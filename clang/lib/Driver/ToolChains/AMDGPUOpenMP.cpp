//===- AMDGPUOpenMP.cpp - AMDGPUOpenMP ToolChain Implementation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUOpenMP.h"
#include "AMDGPU.h"
#include "CommonArgs.h"
#include "ToolChains/ROCm.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetParser.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

#if defined(_WIN32) || defined(_WIN64)
#define NULL_FILE "nul"
#else
#define NULL_FILE "/dev/null"
#endif

namespace {

static void addBCLib(const Driver &D, const ArgList &Args,
                     ArgStringList &CmdArgs, ArgStringList LibraryPaths,
                     StringRef BCName, bool postClangLink) {
  StringRef FullName;
  for (std::string LibraryPath : LibraryPaths) {
    SmallString<128> Path(LibraryPath);
    llvm::sys::path::append(Path, BCName);
    FullName = Path;
    if (llvm::sys::fs::exists(FullName)) {
      if (postClangLink)
        CmdArgs.push_back("-mlink-builtin-bitcode");
      CmdArgs.push_back(Args.MakeArgString(FullName));
      return;
    }
  }
  D.Diag(diag::err_drv_no_such_file) << BCName;
}

static const char *getOutputFileName(Compilation &C, StringRef Base,
                                     const char *Postfix,
                                     const char *Extension) {
  const char *OutputFileName;
  if (C.getDriver().isSaveTempsEnabled()) {
    OutputFileName =
        C.getArgs().MakeArgString(Base.str() + Postfix + "." + Extension);
  } else {
    std::string TmpName =
        C.getDriver().GetTemporaryPath(Base.str() + Postfix, Extension);
    OutputFileName = C.addTempFile(C.getArgs().MakeArgString(TmpName));
  }
  return OutputFileName;
}

static void addLLCOptArg(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs,
			 bool IsLlc = false) {
  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    StringRef OOpt = "0";
    if (A->getOption().matches(options::OPT_O4) ||
        A->getOption().matches(options::OPT_Ofast))
      OOpt = "3";
    else if (A->getOption().matches(options::OPT_O0))
      OOpt = "0";
    else if (A->getOption().matches(options::OPT_O)) {
      // Clang and opt support -Os/-Oz; llc only supports -O0, -O1, -O2 and -O3
      // so we map -Os/-Oz to -O2.
      // Only clang supports -Og, and maps it to -O1.
      // We map anything else to -O2.
      OOpt = llvm::StringSwitch<const char *>(A->getValue())
                 .Case("1", "1")
                 .Case("2", "2")
                 .Case("3", "3")
                 .Case("s", IsLlc ? "2" : "s")
                 .Case("z", IsLlc ? "2" : "z")
                 .Case("g", "1")
                 .Default("0");
    }
    CmdArgs.push_back(Args.MakeArgString("-O" + OOpt));
  }
}

static bool checkSystemForAMDGPU(const ArgList &Args, const AMDGPUToolChain &TC,
                                 std::string &GPUArch) {
  if (auto Err = TC.getSystemGPUArch(Args, GPUArch)) {
    std::string ErrMsg =
        llvm::formatv("{0}", llvm::fmt_consume(std::move(Err)));
    TC.getDriver().Diag(diag::err_drv_undetermined_amdgpu_arch) << ErrMsg;
    return false;
  }

  return true;
}
} // namespace

// OpenMP needs a custom link tool to build select statement
const char *AMDGCN::OpenMPLinker::constructOmpExtraCmds(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const ArgList &Args, StringRef SubArchName,
    StringRef OutputFilePrefix) const {
  ArgStringList CmdArgs;

  std::string TmpName;
  TmpName = C.getDriver().isSaveTempsEnabled()
                ? OutputFilePrefix.str() + "-select.bc"
                : C.getDriver().GetTemporaryPath(
                      OutputFilePrefix.str() + "-select", "bc");
  const char *OutputFileName =
      C.addTempFile(C.getArgs().MakeArgString(TmpName));
  // CmdArgs.push_back("-v");
  llvm::SmallVector<std::string, 10> BCLibs;
  for (const auto &II : Inputs) {
    if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
  }

  ArgStringList LibraryPaths;

  // If device debugging turned on, get bc files from lib-debug dir
  std::string lib_debug_path = FindDebugInLibraryPath();
  if (!lib_debug_path.empty()) {
    LibraryPaths.push_back(Args.MakeArgString(lib_debug_path + "/libdevice"));
    LibraryPaths.push_back(Args.MakeArgString(lib_debug_path));
  }

  addDirectoryList(Args, LibraryPaths, "", "HIP_DEVICE_LIB_PATH");

  // Add compiler path libdevice last as lowest priority search
  LibraryPaths.push_back(
      Args.MakeArgString(C.getDriver().Dir + "/../amdgcn/bitcode"));
  LibraryPaths.push_back(
      Args.MakeArgString(C.getDriver().Dir + "/../../amdgcn/bitcode"));
  LibraryPaths.push_back(
      Args.MakeArgString(C.getDriver().Dir + "/../lib/libdevice"));
  LibraryPaths.push_back(Args.MakeArgString(C.getDriver().Dir + "/../lib"));
  LibraryPaths.push_back(
      Args.MakeArgString(C.getDriver().Dir + "/../../lib/libdevice"));
  LibraryPaths.push_back(Args.MakeArgString(C.getDriver().Dir + "/../../lib"));

  // Add bitcode library in --hip-device-lib.
  for (auto Lib : Args.getAllArgValues(options::OPT_hip_device_lib_EQ)) {
    BCLibs.push_back(Args.MakeArgString(Lib));
  }

  // Add libm for Fortran.
  if (C.getDriver().IsFlangMode()) {
    BCLibs.push_back(Args.MakeArgString("libm-amdgcn-" + SubArchName + ".bc"));
    if (Args.hasArg(options::OPT_cl_finite_math_only))
      BCLibs.push_back(Args.MakeArgString("oclc_finite_only_on.bc"));
    else
      BCLibs.push_back(Args.MakeArgString("oclc_finite_only_off.bc"));
  }

  llvm::StringRef WaveFrontSizeBC;
  std::string GFXVersion = SubArchName.drop_front(3).str();
  if (stoi(GFXVersion) < 1000)
    WaveFrontSizeBC = "oclc_wavefrontsize64_on.bc";
  else
    WaveFrontSizeBC = "oclc_wavefrontsize64_off.bc";

  // FIXME: remove double link of hip aompextras, ockl, and WaveFrontSizeBC
  if (Args.hasArg(options::OPT_cuda_device_only))
    BCLibs.append(
        {Args.MakeArgString("libomptarget-amdgcn-" + SubArchName + ".bc"),
         "hip.bc", "ockl.bc",
         std::string(WaveFrontSizeBC)});
  else {
    BCLibs.append(
        {Args.MakeArgString("libomptarget-amdgcn-" + SubArchName + ".bc"),
         Args.MakeArgString("libaompextras-amdgcn-" + SubArchName + ".bc"),
         "hip.bc", "ockl.bc",
         Args.MakeArgString("libbc-hostrpc-amdgcn.a"),
         std::string(WaveFrontSizeBC)});
  }

  if (Args.hasFlag(options::OPT_fgpu_sanitize,
                   options::OPT_fno_gpu_sanitize, false)) {
    BCLibs.push_back("asanrtl.bc");
  }

  for (auto Lib : BCLibs)
    addBCLib(C.getDriver(), Args, CmdArgs, LibraryPaths, Lib,
             /* PostClang Link? */ false);

  // This will find .a and .bc files that match naming convention.
  AddStaticDeviceLibsLinking(C, *this, JA, Inputs, Args, CmdArgs, "amdgcn",
                      SubArchName,
                      /* bitcode SDL?*/ true,
                      /* PostClang Link? */ false);

  // Get the environment variable ROCM_SELECT_ARGS and add to select-link.
  Optional<std::string> OptEnv = llvm::sys::Process::GetEnv("ROCM_SELECT_ARGS");
  if (OptEnv.hasValue()) {
    SmallVector<StringRef, 8> Envs;
    SplitString(OptEnv.getValue(), Envs);
    for (StringRef Env : Envs)
      CmdArgs.push_back(Args.MakeArgString(Env.trim()));
  }

  CmdArgs.push_back("-o");
  CmdArgs.push_back(OutputFileName);
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(),
      Args.MakeArgString(C.getDriver().Dir + "/clang-build-select-link"),
      CmdArgs, Inputs, InputInfo(&JA, Args.MakeArgString(OutputFileName))));

  return OutputFileName;
}

const char *AMDGCN::OpenMPLinker::constructLLVMLinkCommand(
  const toolchains::AMDGPUOpenMPToolChain &AMDGPUOpenMPTC, Compilation &C,
  const JobAction &JA, const InputInfoList &Inputs,
  const llvm::opt::ArgList &Args, llvm::StringRef SubArchName,
    StringRef OutputFilePrefix) const {
  ArgStringList CmdArgs;

  bool DoOverride = JA.getOffloadingDeviceKind() == Action::OFK_OpenMP;
  StringRef overrideInputsFile =
      DoOverride
          ? constructOmpExtraCmds(C, JA, Inputs, Args, SubArchName,
			          OutputFilePrefix)
          : "";

  // Add the input bc's created by compile step.
  if (overrideInputsFile.empty()) {
    for (const auto &II : Inputs)
      if (II.isFilename())
        CmdArgs.push_back(II.getFilename());
  } else
    CmdArgs.push_back(Args.MakeArgString(overrideInputsFile));

  // for OpenMP, we already did this in clang-build-select-link
  if (JA.getOffloadingDeviceKind() != Action::OFK_OpenMP)
    AddStaticDeviceLibsLinking(C, *this, JA, Inputs, Args, CmdArgs, "amdgcn",
                        SubArchName,
                        /* bitcode SDL?*/ true,
                        /* PostClang Link? */ false);

  // Get the environment variable ROCM_LINK_ARGS and add to llvm-link.
  Optional<std::string> OptEnv = llvm::sys::Process::GetEnv("ROCM_LINK_ARGS");
  if (OptEnv.hasValue()) {
    SmallVector<StringRef, 8> Envs;
    SplitString(OptEnv.getValue(), Envs);
    for (StringRef Env : Envs)
      CmdArgs.push_back(Args.MakeArgString(Env.trim()));
  }

  if (Args.hasArg(options::OPT_l)) {
    auto Lm = Args.getAllArgValues(options::OPT_l);
    bool HasLibm = false;
    for (auto &Lib : Lm) {
      if (Lib == "m") {
        HasLibm = true;
        break;
      }
    }

    if (HasLibm) {
      SmallVector<std::string, 12> BCLibs =
          AMDGPUOpenMPTC.getCommonDeviceLibNames(Args, SubArchName.str());
      llvm::for_each(BCLibs, [&](StringRef BCFile) {
        CmdArgs.push_back(Args.MakeArgString(BCFile));
      });
    }
  }
  // Add an intermediate output file.
  CmdArgs.push_back("-o");
  auto OutputFileName = getOutputFileName(C, OutputFilePrefix, "-linked", "bc");
  CmdArgs.push_back(OutputFileName);
  const char *Exec =
      Args.MakeArgString(getToolChain().GetProgramPath("llvm-link"));
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Exec, CmdArgs, Inputs,
      InputInfo(&JA, Args.MakeArgString(OutputFileName))));
  return OutputFileName;
}

const char *AMDGCN::OpenMPLinker::constructOptCommand(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const llvm::opt::ArgList &Args, llvm::StringRef SubArchName,
    llvm::StringRef OutputFilePrefix, const char *InputFileName) const {
  // Construct opt command.
  ArgStringList OptArgs;
  // The input to opt is the output from llvm-link.
  OptArgs.push_back(InputFileName);
  // Pass optimization arg to opt.
  addLLCOptArg(Args, OptArgs);
  OptArgs.push_back("-mtriple=amdgcn-amd-amdhsa");
  OptArgs.push_back(Args.MakeArgString("-mcpu=" + SubArchName));

  // Get the environment variable ROCM_OPT_ARGS and add to opt.
  Optional<std::string> OptEnv = llvm::sys::Process::GetEnv("ROCM_OPT_ARGS");
  if (OptEnv.hasValue()) {
    SmallVector<StringRef, 8> Envs;
    SplitString(OptEnv.getValue(), Envs);
    for (StringRef Env : Envs)
      OptArgs.push_back(Args.MakeArgString(Env.trim()));
  }

  for (const Arg *A : Args.filtered(options::OPT_mllvm)) {
    OptArgs.push_back(A->getValue(0));
  }

  OptArgs.push_back("-o");
  auto OutputFileName =
      getOutputFileName(C, OutputFilePrefix, "-optimized", "bc");
  OptArgs.push_back(OutputFileName);
  const char *OptExec =
      Args.MakeArgString(getToolChain().GetProgramPath("opt"));
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), OptExec, OptArgs, Inputs,
      InputInfo(&JA, Args.MakeArgString(OutputFileName))));
  return OutputFileName;
}

const char *AMDGCN::OpenMPLinker::constructLlcCommand(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const llvm::opt::ArgList &Args, llvm::StringRef SubArchName,
    llvm::StringRef OutputFilePrefix, const char *InputFileName,
    bool OutputIsAsm) const {
  // Construct llc command.
  ArgStringList LlcArgs;
  // The input to llc is the output from opt.
  LlcArgs.push_back(InputFileName);
  // Pass optimization arg to llc.
  addLLCOptArg(Args, LlcArgs, /*IsLlc=*/true);
  LlcArgs.push_back("-mtriple=amdgcn-amd-amdhsa");
  LlcArgs.push_back(Args.MakeArgString("-mcpu=" + SubArchName));
  LlcArgs.push_back(
      Args.MakeArgString(Twine("-filetype=") + (OutputIsAsm ? "asm" : "obj")));

  // Add the object code version. Example: -mcode-object-version=3
  unsigned ObjCodeVer =
    reinterpret_cast<const AMDGPUToolChain &>(getToolChain())
	  .GetCodeObjectVersion();
  LlcArgs.push_back(
    Args.MakeArgString(Twine("--amdhsa-code-object-version=") +
	               Twine(ObjCodeVer)));

  // Get the environment variable ROCM_LLC_ARGS and add to llc.
  Optional<std::string> OptEnv = llvm::sys::Process::GetEnv("ROCM_LLC_ARGS");
  if (OptEnv.hasValue()) {
    SmallVector<StringRef, 8> Envs;
    SplitString(OptEnv.getValue(), Envs);
    for (StringRef Env : Envs)
      LlcArgs.push_back(Args.MakeArgString(Env.trim()));
  }

  // Extract all the -m options
  std::vector<llvm::StringRef> Features;
  handleTargetFeaturesGroup(
    Args, Features, options::OPT_m_amdgpu_Features_Group);

  // Add features to mattr such as xnack
  std::string MAttrString = "-mattr=";
  for(auto OneFeature : Features) {
    MAttrString.append(Args.MakeArgString(OneFeature));
    if (OneFeature != Features.back())
      MAttrString.append(",");
  }
  if(!Features.empty())
    LlcArgs.push_back(Args.MakeArgString(MAttrString));

  for (const Arg *A : Args.filtered(options::OPT_mllvm)) {
    LlcArgs.push_back(A->getValue(0));
  }

  // Add output filename
  LlcArgs.push_back("-o");
  const char *LlcOutputFile =
      getOutputFileName(C, OutputFilePrefix, "", OutputIsAsm ? "s" : "o");
  LlcArgs.push_back(LlcOutputFile);
  const char *Llc = Args.MakeArgString(getToolChain().GetProgramPath("llc"));
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Llc, LlcArgs, Inputs,
      InputInfo(&JA, Args.MakeArgString(LlcOutputFile))));
  return LlcOutputFile;
}

void AMDGCN::OpenMPLinker::constructLldCommand(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const InputInfo &Output, const llvm::opt::ArgList &Args,
    const char *InputFileName) const {
  // Construct lld command.
  // The output from ld.lld is an HSA code object file.
  ArgStringList LldArgs{"-flavor",    "gnu", "--no-undefined",
                        "-shared",    "-o",  Output.getFilename(),
                        InputFileName};

  // Get the environment variable ROCM_LLD_ARGS and add to lld.
  Optional<std::string> OptEnv = llvm::sys::Process::GetEnv("ROCM_LLD_ARGS");
  if (OptEnv.hasValue()) {
    SmallVector<StringRef, 8> Envs;
    SplitString(OptEnv.getValue(), Envs);
    for (StringRef Env : Envs)
      LldArgs.push_back(Args.MakeArgString(Env.trim()));
  }

  const char *Lld = Args.MakeArgString(getToolChain().GetProgramPath("lld"));
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Lld, LldArgs, Inputs,
      InputInfo(&JA, Args.MakeArgString(Output.getFilename()))));
}

// For amdgcn the inputs of the linker job are device bitcode and output is
// object file. It calls llvm-link, opt, llc, then lld steps.
void AMDGCN::OpenMPLinker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {

  if (JA.getType() == types::TY_HIP_FATBIN)
    return constructHIPFatbinCommand(C, JA, Output.getFilename(), Inputs, Args, *this);
  const ToolChain &TC = getToolChain();
  assert(getToolChain().getTriple().isAMDGCN() && "Unsupported target");

  const toolchains::AMDGPUOpenMPToolChain &AMDGPUOpenMPTC =
      static_cast<const toolchains::AMDGPUOpenMPToolChain &>(TC);

  std::string GPUArch = Args.getLastArgValue(options::OPT_march_EQ).str();
  if (GPUArch.empty()) {
    if (!checkSystemForAMDGPU(Args, AMDGPUOpenMPTC, GPUArch))
      return;
  }

  // Prefix for temporary file name.
  std::string Prefix;
  for (const auto &II : Inputs)
    if (II.isFilename())
      Prefix = llvm::sys::path::stem(II.getFilename()).str() + "-" + GPUArch;
  assert(Prefix.length() && "no linker inputs are files ");

  // Each command outputs different files.
  const char *LLVMLinkCommand =
      constructLLVMLinkCommand(AMDGPUOpenMPTC, C, JA, Inputs, Args,  GPUArch.c_str(), Prefix);
  const char *OptCommand = constructOptCommand(C, JA, Inputs, Args,
		                               GPUArch.c_str(),
                                               Prefix, LLVMLinkCommand);
  // Produce readable assembly if save-temps is enabled.
  if (C.getDriver().isSaveTempsEnabled())
    constructLlcCommand(C, JA, Inputs, Args, GPUArch.c_str(), Prefix, OptCommand,
                        /*OutputIsAsm=*/true);
  const char *LlcCommand =
      constructLlcCommand(C, JA, Inputs, Args, GPUArch.c_str(), Prefix,
		          OptCommand);
  constructLldCommand(C, JA, Inputs, Output, Args, LlcCommand);
}

AMDGPUOpenMPToolChain::AMDGPUOpenMPToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args,
                             const Action::OffloadKind OK)
    : ROCMToolChain(D, Triple, Args), HostTC(HostTC), OK(OK) {
  // Lookup binaries into the driver directory, this is used to
  // discover the clang-offload-bundler executable.
  getProgramPaths().push_back(getDriver().Dir);
}

void AMDGPUOpenMPToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);
  std::string GPUArch = DriverArgs.getLastArgValue(options::OPT_march_EQ).str();
  if (GPUArch.empty()) {
    if (!checkSystemForAMDGPU(DriverArgs, *this, GPUArch))
      return;
  }
  assert((DeviceOffloadingKind == Action::OFK_HIP ||
          DeviceOffloadingKind == Action::OFK_OpenMP) &&
         "Only HIP offloading kinds are supported for GPUs.");
  auto Kind = llvm::AMDGPU::parseArchAMDGCN(GPUArch);
  const StringRef CanonArch = llvm::AMDGPU::getArchNameAMDGCN(Kind);

  CC1Args.push_back("-target-cpu");
  CC1Args.push_back(DriverArgs.MakeArgStringRef(GPUArch));
  CC1Args.push_back("-fcuda-is-device");

  if (DriverArgs.hasFlag(options::OPT_fcuda_approx_transcendentals,
                         options::OPT_fno_cuda_approx_transcendentals, false))
    CC1Args.push_back("-fcuda-approx-transcendentals");

  if (DriverArgs.hasFlag(options::OPT_fgpu_rdc, options::OPT_fno_gpu_rdc,
                         false))
    CC1Args.push_back("-fgpu-rdc");

  if (DriverArgs.hasArg(options::OPT_S) && DriverArgs.hasArg(options::OPT_emit_llvm))
    CC1Args.push_back("-S");

  StringRef MaxThreadsPerBlock =
    DriverArgs.getLastArgValue(options::OPT_gpu_max_threads_per_block_EQ);
  if (!MaxThreadsPerBlock.empty()) {
    std::string ArgStr =
      std::string("--gpu-max-threads-per-block=") + MaxThreadsPerBlock.str();
    CC1Args.push_back(DriverArgs.MakeArgStringRef(ArgStr));
  }

  if (DriverArgs.hasFlag(options::OPT_fgpu_allow_device_init,
                         options::OPT_fno_gpu_allow_device_init, false))
    CC1Args.push_back("-fgpu-allow-device-init");

  CC1Args.push_back("-fcuda-allow-variadic-functions");

  // Default to "hidden" visibility, as object level linking will not be
  // supported for the foreseeable future.
  if (!DriverArgs.hasArg(options::OPT_fvisibility_EQ,
                         options::OPT_fvisibility_ms_compat) &&
      DeviceOffloadingKind != Action::OFK_OpenMP) {
    CC1Args.append({"-fvisibility", "hidden"});
    CC1Args.push_back("-fapply-global-visibility-to-externs");
  }

  if (DriverArgs.hasArg(options::OPT_nogpulib))
    return;
  ArgStringList LibraryPaths;

  // Find in --hip-device-lib-path and HIP_LIBRARY_PATH.
  for (auto Path :
       RocmInstallation.getRocmDeviceLibPathArg())
    LibraryPaths.push_back(DriverArgs.MakeArgString(Path));

  addDirectoryList(DriverArgs, LibraryPaths, "", "HIP_DEVICE_LIB_PATH");

  // Maintain compatability with --hip-device-lib.
  auto BCLibs = DriverArgs.getAllArgValues(options::OPT_hip_device_lib_EQ);
  if (!BCLibs.empty()) {
    for (auto Lib : BCLibs)
      addBCLib(getDriver(), DriverArgs, CC1Args, LibraryPaths, Lib,
               /* PostClang Link? */ true);

  } else {
    if (!RocmInstallation.hasDeviceLibrary()) {
      getDriver().Diag(diag::err_drv_no_rocm_device_lib) << 0;
      return;
    }

    // If device debugging turned on, add specially built bc files
    std::string lib_debug_path = FindDebugInLibraryPath();
    if (!lib_debug_path.empty()) {
      LibraryPaths.push_back(
          DriverArgs.MakeArgString(lib_debug_path + "/libdevice"));
      LibraryPaths.push_back(DriverArgs.MakeArgString(lib_debug_path));
    }

    // Add compiler path libdevice last as lowest priority search
    LibraryPaths.push_back(
        DriverArgs.MakeArgString(getDriver().Dir + "/../amdgcn/bitcode"));
    LibraryPaths.push_back(
        DriverArgs.MakeArgString(getDriver().Dir + "/../../amdgcn/bitcode"));
    LibraryPaths.push_back(
        DriverArgs.MakeArgString(getDriver().Dir + "/../lib/libdevice"));
    LibraryPaths.push_back(
        DriverArgs.MakeArgString(getDriver().Dir + "/../lib"));
    LibraryPaths.push_back(
        DriverArgs.MakeArgString(getDriver().Dir + "/../../lib/libdevice"));
    LibraryPaths.push_back(
	DriverArgs.MakeArgString(getDriver().Dir + "/../../lib"));

    std::string LibDeviceFile = RocmInstallation.getLibDeviceFile(CanonArch);
    if (LibDeviceFile.empty()) {
      getDriver().Diag(diag::err_drv_no_rocm_device_lib) << 1 << GPUArch;
      return;
    }

    // If --hip-device-lib is not set, add the default bitcode libraries.
    // TODO: There are way too many flags that change this. Do we need to check
    // them all?
    bool DAZ = DriverArgs.hasFlag(options::OPT_fcuda_flush_denormals_to_zero,
                                  options::OPT_fno_cuda_flush_denormals_to_zero,
                                  getDefaultDenormsAreZeroForTarget(Kind));
    // TODO: Check standard C++ flags?
    bool FiniteOnly = false;
    bool UnsafeMathOpt = false;
    bool FastRelaxedMath = false;
    bool CorrectSqrt = true;
    bool Wave64 = isWave64(DriverArgs, Kind);

    // Add the HIP specific bitcode library.
    llvm::SmallVector<std::string, 12> BCLibs;
    BCLibs.push_back(RocmInstallation.getHIPPath().str());

    // Add the generic set of libraries.
    BCLibs.append(RocmInstallation.getCommonBitcodeLibs(
        DriverArgs, LibDeviceFile, Wave64, DAZ, FiniteOnly, UnsafeMathOpt,
        FastRelaxedMath, CorrectSqrt));

    llvm::for_each(BCLibs, [&](StringRef BCFile) {
      CC1Args.push_back("-mlink-builtin-bitcode");
      CC1Args.push_back(DriverArgs.MakeArgString(BCFile));
    });
  }

  if (!DriverArgs.hasArg(options::OPT_l))
    return;

  auto Lm = DriverArgs.getAllArgValues(options::OPT_l);
  bool HasLibm = false;
  for (auto &Lib : Lm) {
    if (Lib == "m") {
      HasLibm = true;
      break;
    }
  }

  if (HasLibm) {
    SmallVector<std::string, 12> BCLibs =
        getCommonDeviceLibNames(DriverArgs, GPUArch);
    llvm::for_each(BCLibs, [&](StringRef BCFile) {
      CC1Args.push_back("-mlink-builtin-bitcode");
      CC1Args.push_back(DriverArgs.MakeArgString(BCFile));
    });
  }
  // Fixme: do we need this?
  //addOpenMPDeviceRTL(getDriver(), DriverArgs, CC1Args, BitcodeSuffix,
  //                   getTriple());
}

llvm::opt::DerivedArgList *AMDGPUOpenMPToolChain::TranslateArgs(
    const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
    Action::OffloadKind DeviceOffloadKind) const {

  DerivedArgList *DAL =
      HostTC.TranslateArgs(Args, BoundArch, DeviceOffloadKind);
  if (!DAL)
    DAL = new DerivedArgList(Args.getBaseArgs());

  const OptTable &Opts = getDriver().getOpts();

  if (DeviceOffloadKind != Action::OFK_OpenMP) {
    for (Arg *A : Args) {
      DAL->append(A);
    }
  }

  if (!BoundArch.empty()) {
    DAL->eraseArg(options::OPT_march_EQ);
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ),
                      BoundArch);
  }

  return DAL;
}

Tool *AMDGPUOpenMPToolChain::buildLinker() const {
  assert(getTriple().isAMDGCN());
  return new tools::AMDGCN::OpenMPLinker(*this);
}

void AMDGPUOpenMPToolChain::addClangWarningOptions(
    ArgStringList &CC1Args) const {
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
AMDGPUOpenMPToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void AMDGPUOpenMPToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  const Driver &D = HostTC.getDriver();
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(D.Dir + "/../include"));
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(D.Dir + "/../../include"));

  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);

  CC1Args.push_back("-internal-isystem");
  SmallString<128> P(HostTC.getDriver().ResourceDir);
  llvm::sys::path::append(P, "include/cuda_wrappers");
  CC1Args.push_back(DriverArgs.MakeArgString(P));
}

/// Convert path list to Fortran frontend argument
static void AddFlangSysIncludeArg(const ArgList &DriverArgs,
                                  ArgStringList &Flang1args,
                                  ToolChain::path_list IncludePathList) {
  std::string ArgValue; // Path argument value

  // Make up argument value consisting of paths separated by colons
  bool first = true;
  for (auto P : IncludePathList) {
    if (first) {
      first = false;
    } else {
      ArgValue += ":";
    }
    ArgValue += P;
  }

  // Add the argument
  Flang1args.push_back("-stdinc");
  Flang1args.push_back(DriverArgs.MakeArgString(ArgValue));
}

/// Currently only adding include dir from install directory
void AMDGPUOpenMPToolChain::AddFlangSystemIncludeArgs(const ArgList &DriverArgs,
                                            ArgStringList &Flang1args) const {
  path_list IncludePathList;
  const Driver &D = getDriver();

  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  {
    SmallString<128> P(D.InstalledDir);
    llvm::sys::path::append(P, "../include");
    IncludePathList.push_back(DriverArgs.MakeArgString(P.str()));
  }

  AddFlangSysIncludeArg(DriverArgs, Flang1args, IncludePathList);
  return;
}


void AMDGPUOpenMPToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                                 ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}

void AMDGPUOpenMPToolChain::AddIAMCUIncludeArgs(const ArgList &Args,
                                                ArgStringList &CC1Args) const {
  HostTC.AddIAMCUIncludeArgs(Args, CC1Args);
}

SanitizerMask AMDGPUOpenMPToolChain::getSupportedSanitizers() const {
  // The AMDGPUOpenMPToolChain only supports sanitizers in the sense that it
  // allows sanitizer arguments on the command line if they are supported by the
  // host toolchain. The AMDGPUOpenMPToolChain will actually ignore any command
  // line arguments for any of these "supported" sanitizers. That means that no
  // sanitization of device code is actually supported at this time.
  //
  // This behavior is necessary because the host and device toolchains
  // invocations often share the command line, so the device toolchain must
  // tolerate flags meant only for the host toolchain.
  return HostTC.getSupportedSanitizers();
}

VersionTuple
AMDGPUOpenMPToolChain::computeMSVCVersion(const Driver *D,
                                          const ArgList &Args) const {
  return HostTC.computeMSVCVersion(D, Args);
}
