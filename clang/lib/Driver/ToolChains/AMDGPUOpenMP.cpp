//===- AMDGPUOpenMP.cpp - AMDGPUOpenMP ToolChain Implementation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
// Notified per clause 4(b) of the license.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUOpenMP.h"
#include "AMDGPU.h"
#include "CommonArgs.h"
#include "HIPUtility.h"
#include "ToolChains/ROCm.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "clang/Driver/Tool.h"
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

const char *AMDGCN::OpenMPLinker::constructLLVMLinkCommand(
    const toolchains::AMDGPUOpenMPToolChain &AMDGPUOpenMPTC, Compilation &C,
    const JobAction &JA, const InputInfoList &Inputs,
    const llvm::opt::ArgList &Args, llvm::StringRef TargetID,
    StringRef OutputFilePrefix) const {

  const char *Exec =
      Args.MakeArgString(getToolChain().GetProgramPath("llvm-link"));
  ArgStringList LastLinkArgs;

  // Look for Static Device Libs (SDLs) in args, and add temp files for
  // the extracted Device-specific Archive Libs (DAL) to inputs
  ArgStringList CmdArgs;
  AddStaticDeviceLibsLinking(C, *this, JA, Inputs, Args, CmdArgs, "amdgcn",
                             TargetID,
                             /* bitcode SDL?*/ true,
                             /* PostClang Link? */ false);

  // Count linking inputs for linking
  int input_count = 0;
  for (const auto &II : Inputs)
    if (II.isFilename())
      input_count++;

  StringRef disable_fn = Args.MakeArgString(
      C.getDriver().Dir + "/../lib/disable_dynamic_devmem.ll");
  // When requested by the user via -fdisable-host-devmem,
  // to avoid host service thread for potential performance concerns,
  // disable host assisted device memory
  // management by providing empty implementation of devmem routine
  // (only available in new device rtl)
  if (llvm::sys::fs::exists(disable_fn) &&
      Args.hasFlag(options::OPT_fdisable_host_devmem,
                   options::OPT_fenable_host_devmem, false) &&
      Args.hasFlag(options::OPT_fopenmp_target_new_runtime,
                   options::OPT_fno_openmp_target_new_runtime, true)) {
    input_count++;
    CmdArgs.push_back(Args.MakeArgString(disable_fn));
  }

  // If more than 1 input or need to link any SDLs, we need a pre-link step.
  if ((input_count > 1) || !CmdArgs.empty()) {
    // ArgStringList CmdArgs;
    for (const auto &II : Inputs)
      if (II.isFilename())
        CmdArgs.push_back(II.getFilename());
    CmdArgs.push_back("-o");
    auto PreLinkFileName =
        getOutputFileName(C, OutputFilePrefix, "-prelinked", "bc");
    CmdArgs.push_back(PreLinkFileName);
    C.addCommand(std::make_unique<Command>(
        JA, *this, ResponseFileSupport::AtFileCurCP(), Exec, CmdArgs, Inputs,
        InputInfo(&JA, Args.MakeArgString(PreLinkFileName))));
    // Output of prelink is only input to last link
    LastLinkArgs.push_back(Args.MakeArgString(PreLinkFileName));
  } else {
    // If only a single input, use it for lastlink input
    for (const auto &II : Inputs)
      if (II.isFilename())
        LastLinkArgs.push_back(II.getFilename());
  }

  // Last link brings in libomptarget and subset of user-option bc files.
  // This link uses --internalize to internalize libomptarget symbols.
  // --internalize ignores the first bc file which came from previous link.

  // Get the environment variable ROCM_LINK_ARGS and add to llvm-link.
  Optional<std::string> OptEnv = llvm::sys::Process::GetEnv("ROCM_LINK_ARGS");
  if (OptEnv.hasValue()) {
    SmallVector<StringRef, 8> Envs;
    SplitString(OptEnv.getValue(), Envs);
    for (StringRef Env : Envs)
      LastLinkArgs.push_back(Args.MakeArgString(Env.trim()));
  }

  LastLinkArgs.push_back(Args.MakeArgString("--internalize"));
  LastLinkArgs.push_back(Args.MakeArgString("--only-needed"));
  StringRef GPUArch =
      getProcessorFromTargetID(getToolChain().getTriple(), TargetID);

  // If device debugging turned on, add specially built bc files
  StringRef libpath = Args.MakeArgString(C.getDriver().Dir + "/../lib");
  std::string lib_debug_path = FindDebugInLibraryPath();
  if (!lib_debug_path.empty())
    libpath = lib_debug_path;

  llvm::SmallVector<std::string, 12> BCLibs;

  if (Args.hasFlag(options::OPT_fgpu_sanitize, options::OPT_fno_gpu_sanitize,
                   true) &&
      AMDGPUOpenMPTC.getSanitizerArgs(Args).needsAsanRt()) {
    std::string AsanRTL(
        AMDGPUOpenMPTC.getRocmInstallationLoc().getAsanRTLPath());
    if (AsanRTL.empty()) {
      AMDGPUOpenMPTC.getDriver().Diag(diag::err_drv_no_asan_rt_lib);
    } else {
      BCLibs.push_back(AsanRTL);
    }
  }

  if (Args.hasFlag(options::OPT_fopenmp_target_new_runtime,
                   options::OPT_fno_openmp_target_new_runtime, true))
    BCLibs.push_back(Args.MakeArgString(libpath + "/libomptarget-new-amdgpu-" +
                                        GPUArch + ".bc"));
  else
    BCLibs.push_back(Args.MakeArgString(libpath + "/libomptarget-amdgcn-" +
                                        GPUArch + ".bc"));

  // Add the generic set of libraries, OpenMP subset only
  BCLibs.append(AMDGPUOpenMPTC.getCommonDeviceLibNames(Args, GPUArch.str(),
                                                       /* isOpenMP=*/true));
  llvm::for_each(BCLibs, [&](StringRef BCFile) {
    LastLinkArgs.push_back(Args.MakeArgString(BCFile));
  });

  LastLinkArgs.push_back("-o");
  auto OutputFileName = getOutputFileName(C, OutputFilePrefix, "-linked", "bc");
  LastLinkArgs.push_back(OutputFileName);
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Exec, LastLinkArgs, Inputs,
      InputInfo(&JA, Args.MakeArgString(OutputFileName))));
  return OutputFileName;
}

const char *AMDGCN::OpenMPLinker::constructOptCommand(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const llvm::opt::ArgList &Args, llvm::StringRef TargetID,
    llvm::StringRef OutputFilePrefix, const char *InputFileName) const {
  // Construct opt command.
  ArgStringList OptArgs;
  // The input to opt is the output from llvm-link.
  OptArgs.push_back(InputFileName);
  StringRef GPUArch =
      getProcessorFromTargetID(getToolChain().getTriple(), TargetID);

  // Pass optimization arg to opt.
  addLLCOptArg(Args, OptArgs);
  OptArgs.push_back("-mtriple=amdgcn-amd-amdhsa");
  OptArgs.push_back(Args.MakeArgString("-mcpu=" + GPUArch));
  // OptArgs.push_back(Args.MakeArgString("-openmp-opt-disable=1"));

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

  auto &TC = getToolChain();
  auto &D = TC.getDriver();
  // Extract all the -m options
  std::vector<llvm::StringRef> Features;
  amdgpu::getAMDGPUTargetFeatures(D, TC.getTriple(), Args, Features,
                                  TargetID.str());

  // Add features to mattr such as xnack
  std::string MAttrString = "-mattr=";
  for (auto OneFeature : Features) {
    MAttrString.append(Args.MakeArgString(OneFeature));
    if (OneFeature != Features.back())
      MAttrString.append(",");
  }
  if (!Features.empty())
    OptArgs.push_back(Args.MakeArgString(MAttrString));

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
    const llvm::opt::ArgList &Args, llvm::StringRef TargetID,
    llvm::StringRef OutputFilePrefix, const char *InputFileName,
    bool OutputIsAsm) const {
  // Construct llc command.
  ArgStringList LlcArgs;
  // The input to llc is the output from opt.
  LlcArgs.push_back(InputFileName);
  // Pass optimization arg to llc.
  addLLCOptArg(Args, LlcArgs, /*IsLlc=*/true);
  LlcArgs.push_back("-mtriple=amdgcn-amd-amdhsa");
  StringRef GPUArch =
      getProcessorFromTargetID(getToolChain().getTriple(), TargetID);
  LlcArgs.push_back(Args.MakeArgString("-mcpu=" + GPUArch));
  LlcArgs.push_back(
      Args.MakeArgString(Twine("-filetype=") + (OutputIsAsm ? "asm" : "obj")));

  // Get the environment variable ROCM_LLC_ARGS and add to llc.
  Optional<std::string> OptEnv = llvm::sys::Process::GetEnv("ROCM_LLC_ARGS");
  if (OptEnv.hasValue()) {
    SmallVector<StringRef, 8> Envs;
    SplitString(OptEnv.getValue(), Envs);
    for (StringRef Env : Envs)
      LlcArgs.push_back(Args.MakeArgString(Env.trim()));
  }

  auto &TC = getToolChain();
  auto &D = TC.getDriver();
  // Extract all the -m options
  std::vector<llvm::StringRef> Features;
  amdgpu::getAMDGPUTargetFeatures(D, TC.getTriple(), Args, Features,
                                  TargetID.str());

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

  if (Arg *A = Args.getLastArgNoClaim(options::OPT_g_Group))
    if (!A->getOption().matches(options::OPT_g0) &&
        !A->getOption().matches(options::OPT_ggdb0))
      LlcArgs.push_back("-amdgpu-spill-cfi-saved-regs");

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
  ArgStringList LldArgs{"-flavor", "gnu", "--no-undefined", "-shared", "-o"};

  auto &TC = getToolChain();
  auto &D = TC.getDriver();
  auto TargetID = TC.getTargetID();
  LldArgs.push_back(Args.MakeArgString(Output.getFilename()));

  LldArgs.push_back(Args.MakeArgString(InputFileName));
  // Get the environment variable ROCM_LLD_ARGS and add to lld.
  Optional<std::string> OptEnv = llvm::sys::Process::GetEnv("ROCM_LLD_ARGS");
  if (OptEnv.hasValue()) {
    SmallVector<StringRef, 8> Envs;
    SplitString(OptEnv.getValue(), Envs);
    for (StringRef Env : Envs)
      LldArgs.push_back(Args.MakeArgString(Env.trim()));
  }

  StringRef GPUArch = getProcessorFromTargetID(TC.getTriple(), TargetID);
  LldArgs.push_back(Args.MakeArgString("-plugin-opt=mcpu=" + GPUArch));

  if (Arg *A = Args.getLastArgNoClaim(options::OPT_g_Group))
    if (!A->getOption().matches(options::OPT_g0) &&
        !A->getOption().matches(options::OPT_ggdb0))
      LldArgs.push_back("-plugin-opt=-amdgpu-spill-cfi-saved-regs");

  // Extract all the -m options
  std::vector<llvm::StringRef> Features;
  amdgpu::getAMDGPUTargetFeatures(D, TC.getTriple(), Args, Features, TargetID);

  // Add features to mattr such as cumode
  std::string MAttrString = "-plugin-opt=-mattr=";
  for (auto OneFeature : unifyTargetFeatures(Features)) {
    MAttrString.append(Args.MakeArgString(OneFeature));
    if (OneFeature != Features.back())
      MAttrString.append(",");
  }
  if (!Features.empty())
    LldArgs.push_back(Args.MakeArgString(MAttrString));

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
    return HIP::constructHIPFatbinCommand(C, JA, Output.getFilename(), Inputs, Args, *this);
  const ToolChain &TC = getToolChain();
  assert(getToolChain().getTriple().isAMDGCN() && "Unsupported target");

  const toolchains::AMDGPUOpenMPToolChain &AMDGPUOpenMPTC =
      static_cast<const toolchains::AMDGPUOpenMPToolChain &>(TC);

  StringRef TargetID = AMDGPUOpenMPTC.getTargetID();
  assert(TargetID.startswith("gfx") && "Unsupported sub arch");

  // Prefix for temporary file name.
  std::string Prefix;
  for (const auto &II : Inputs)
    if (II.isFilename())
      Prefix =
          llvm::sys::path::stem(II.getFilename()).str() + "-" + TargetID.str();
  assert(Prefix.length() && "no linker inputs are files ");

  // Each command outputs different files.
  const char *LLVMLinkCommand = constructLLVMLinkCommand(
      AMDGPUOpenMPTC, C, JA, Inputs, Args, TargetID, Prefix);
  const char *OptCommand = constructOptCommand(C, JA, Inputs, Args, TargetID,
                                               Prefix, LLVMLinkCommand);
  // Produce readable assembly if save-temps is enabled.
  if (C.getDriver().isSaveTempsEnabled())
    constructLlcCommand(C, JA, Inputs, Args, TargetID, Prefix, OptCommand,
                        /*OutputIsAsm=*/true);
  const char *LlcCommand =
      constructLlcCommand(C, JA, Inputs, Args, TargetID, Prefix, OptCommand);
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

AMDGPUOpenMPToolChain::AMDGPUOpenMPToolChain(const Driver &D,
                                             const llvm::Triple &Triple,
                                             const ToolChain &HostTC,
                                             const ArgList &Args,
                                             const Action::OffloadKind OK,
                                             const std::string TargetID)
    : ROCMToolChain(D, Triple, Args), HostTC(HostTC), OK(OK) {
  // Lookup binaries into the driver directory, this is used to
  // discover the clang-offload-bundler executable.
  getProgramPaths().push_back(getDriver().Dir);
  this->TargetID = std::move(TargetID);
}

void AMDGPUOpenMPToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);

  std::string TargetIDStr = getTargetID().str();
  if (TargetIDStr.empty()) {
    if (!checkSystemForAMDGPU(DriverArgs, *this, TargetIDStr))
      return;
  }
  StringRef TargetID = StringRef(TargetIDStr);
  assert((DeviceOffloadingKind == Action::OFK_HIP ||
          DeviceOffloadingKind == Action::OFK_OpenMP) &&
         "Only HIP offloading kinds are supported for GPUs.");

  CC1Args.push_back("-target-cpu");
  StringRef GPUArch = getProcessorFromTargetID(getTriple(), TargetID);
  CC1Args.push_back(DriverArgs.MakeArgStringRef(GPUArch));

  // Extract all the -m options
  std::vector<llvm::StringRef> Features;
  amdgpu::getAMDGPUTargetFeatures(getDriver(), getTriple(), DriverArgs,
                                  Features, TargetIDStr);

  for (auto OneFeature : unifyTargetFeatures(Features)) {
    CC1Args.push_back("-target-feature");
    CC1Args.push_back(OneFeature.data());
  }

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

  // Link the bitcode library late if we're using device LTO.
  if (getDriver().isUsingLTO(/* IsOffload */ true))
    return;

  std::string BitcodeSuffix;
  if (DriverArgs.hasFlag(options::OPT_fopenmp_target_new_runtime,
                         options::OPT_fno_openmp_target_new_runtime, true))
    BitcodeSuffix = llvm::Twine("new-amdgpu-" + GPUArch).str();
  else
    BitcodeSuffix = llvm::Twine("amdgcn-" + GPUArch).str();

  addDirectoryList(DriverArgs, LibraryPaths, "", "HIP_DEVICE_LIB_PATH");

  // Maintain compatability with --hip-device-lib.
  auto BCLibs = DriverArgs.getAllArgValues(options::OPT_hip_device_lib_EQ);
  if (!BCLibs.empty())
    for (auto Lib : BCLibs)
      addBCLib(getDriver(), DriverArgs, CC1Args, LibraryPaths, Lib,
               /* PostClang Link? */ true);
}

llvm::opt::DerivedArgList *AMDGPUOpenMPToolChain::TranslateArgs(
    const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
    Action::OffloadKind DeviceOffloadKind) const {

  DerivedArgList *DAL =
      HostTC.TranslateArgs(Args, BoundArch, DeviceOffloadKind);
  if (!DAL)
    DAL = new DerivedArgList(Args.getBaseArgs());

  const OptTable &Opts = getDriver().getOpts();

  if (DeviceOffloadKind == Action::OFK_OpenMP) {
    for (Arg *A : Args) {
      if (!shouldSkipSanitizeOption(*this, Args, BoundArch, A) &&
          !llvm::is_contained(*DAL, A))
        DAL->append(A);
    }

    if (!DAL->hasArg(options::OPT_march_EQ)) {
      std::string Arch = BoundArch.str();
      if (Arch.empty()) {
        Arch = getTargetID().str(); // arch may have come from --Offload-Arch=
        if (Arch.empty())
          checkSystemForAMDGPU(Args, *this, Arch);
      }
      DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ), Arch);
    }

    return DAL;
  }

  for (Arg *A : Args) {
    DAL->append(A);
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
