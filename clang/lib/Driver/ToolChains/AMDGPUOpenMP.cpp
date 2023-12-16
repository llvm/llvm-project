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
#include "llvm/ADT/StringExtras.h"    
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/TargetParser/TargetParser.h"
#include "clang/Config/config.h"

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

static bool checkSystemForAMDGPU(const ArgList &Args, const AMDGPUToolChain &TC,
                                 std::string &GPUArch) {
  auto CheckError = [&](llvm::Error Err) -> bool {
    std::string ErrMsg =
        llvm::formatv("{0}", llvm::fmt_consume(std::move(Err)));
    TC.getDriver().Diag(diag::err_drv_undetermined_gpu_arch)
        << llvm::Triple::getArchTypeName(TC.getArch()) << ErrMsg << "-march";
    return false;
  };

  auto ArchsOrErr = TC.getSystemGPUArchs(Args);
  if (!ArchsOrErr)
    return CheckError(ArchsOrErr.takeError());

  if (ArchsOrErr->size() > 1)
    if (!llvm::all_equal(*ArchsOrErr))
      return CheckError(llvm::createStringError(
          std::error_code(), "Multiple AMD GPUs found with different archs"));

  GPUArch = ArchsOrErr->front();
  return true;
}

static void addOptLevelArg(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs, bool IsLlc) {
  StringRef OOpt = "2"; // Default if no user command line specification
  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
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
  } else {
    // Nothing in the O_Group
    if (isTargetFastUsed(Args))
      OOpt = "3";
  }
  // To remove unreferenced internalized functions, add globaldce pass to O0
  if (OOpt.equals("0") && !IsLlc)
    CmdArgs.push_back(Args.MakeArgString("-passes=default<O0>,globaldce"));
  else
    CmdArgs.push_back(Args.MakeArgString("-O" + OOpt));
}

static void addAMDTargetArgs(Compilation &C, const llvm::opt::ArgList &Args,
                             llvm::opt::ArgStringList &CmdArgs, bool IsLlc) {
  unsigned CodeObjVer =
      getOrCheckAMDGPUCodeObjectVersion(C.getDriver(), C.getArgs(), true);
  if (CodeObjVer)
    CmdArgs.push_back(Args.MakeArgString(
        Twine("--amdhsa-code-object-version=") + Twine(CodeObjVer)));

  // Pass optimization arg to llc.
  addOptLevelArg(Args, CmdArgs, /*IsLlc=*/IsLlc);
  CmdArgs.push_back("-mtriple=amdgcn-amd-amdhsa");
}

static void addROCmEnvArgs(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs,
                           const char *ROCmEnvVarName) {
  // Get the environment variable and add to llc.
  std::optional<std::string> OptEnv =
      llvm::sys::Process::GetEnv(ROCmEnvVarName);
  if (OptEnv.has_value()) {
    SmallVector<StringRef, 8> Envs;
    SplitString(OptEnv.value(), Envs);
    for (StringRef Env : Envs)
      CmdArgs.push_back(Args.MakeArgString(Env.trim()));
  }
}

static void addCommonArgs(Compilation &C, const llvm::opt::ArgList &Args,
                          llvm::opt::ArgStringList &CmdArgs,
                          const llvm::Triple &Triple, llvm::StringRef TargetID,
                          const char *InputFileName, const char *ROCmEnvVarName,
                          bool isLld = false) {
  CmdArgs.push_back(InputFileName);

  StringRef GPUArch = getProcessorFromTargetID(Triple, TargetID);
  CmdArgs.push_back(
      Args.MakeArgString((isLld ? "-plugin-opt=mcpu=" : "-mcpu=") + GPUArch));

  // Get the environment variable and add command args
  addROCmEnvArgs(Args, CmdArgs, ROCmEnvVarName);

  // Extract all the -m options
  std::vector<llvm::StringRef> Features;
  amdgpu::getAMDGPUTargetFeatures(C.getDriver(), Triple, Args, Features,
                                  TargetID.str());

  // Add features to mattr such as xnack
  std::string MAttrString = isLld ? "-plugin-opt=-mattr=" : "-mattr=";
  for (auto OneFeature : Features) {
    MAttrString.append(Args.MakeArgString(OneFeature));
    if (OneFeature != Features.back())
      MAttrString.append(",");
  }
  if (!Features.empty())
    CmdArgs.push_back(Args.MakeArgString(MAttrString));

  if (!isLld)
    for (const Arg *A : Args.filtered(options::OPT_mllvm))
      CmdArgs.push_back(A->getValue(0));
}
} // namespace

const char *amdgpu::dlr::getCbslCommandArgs(
    Compilation &C, const llvm::opt::ArgList &Args,
    llvm::opt::ArgStringList &CbslArgs,
    const SmallVectorImpl<std::string> &InputFileNames,
    llvm::StringRef OutputFilePrefix) {
  StringRef disable_fn = Args.MakeArgString(
      C.getDriver().Dir + "/../lib/disable_dynamic_devmem.ll");

  // When requested by the user via -fdisable-host-devmem,
  // to avoid host service thread for potential performance concerns,
  // disable host assisted device memory
  // management by providing empty implementation of devmem routine
  // (only available in new device rtl)
  if (llvm::sys::fs::exists(disable_fn) &&
      Args.hasFlag(options::OPT_fdisable_host_devmem,
                   options::OPT_fenable_host_devmem, false))
    CbslArgs.push_back(Args.MakeArgString(disable_fn));

  for (const auto &II : InputFileNames)
    CbslArgs.push_back(Args.MakeArgString(II));

  // Get the environment variable ROCM_CBSL_ARGS and add to
  // clang-build-select-link.
  addROCmEnvArgs(Args, CbslArgs, "ROCM_CBSL_ARGS");

  CbslArgs.push_back("-o");
  auto PreLinkFileName =
      getOutputFileName(C, OutputFilePrefix, "-prelinked", "bc");
  CbslArgs.push_back(PreLinkFileName);
  return PreLinkFileName;
}

const char *amdgpu::dlr::getLinkCommandArgs(
    Compilation &C, const llvm::opt::ArgList &Args,
    llvm::opt::ArgStringList &LastLinkArgs, const ToolChain &TC,
    const llvm::Triple &Triple, llvm::StringRef TargetID,
    llvm::StringRef OutputFilePrefix, const char *InputFileName,
    const RocmInstallationDetector &RocmInstallation) {
  LastLinkArgs.push_back(Args.MakeArgString(InputFileName));

  // Get the environment variable ROCM_LINK_ARGS and add to llvm-link.
  addROCmEnvArgs(Args, LastLinkArgs, "ROCM_LINK_ARGS");

  // Last link brings in libomptarget and subset of user-option bc files.
  // This link uses --internalize to internalize libomptarget symbols.
  // --internalize ignores the first bc file which came from previous link.
  LastLinkArgs.push_back(Args.MakeArgString("--internalize"));
  LastLinkArgs.push_back(Args.MakeArgString("--only-needed"));

  std::string LibSuffix = "lib";
  if (TC.getSanitizerArgs(Args).needsAsanRt())
    LibSuffix.append("/asan");
  if (Arg *A = Args.getLastArg(options::OPT_fopenmp_runtimelib_EQ)) {
    LibSuffix = A->getValue();
    if (TC.getSanitizerArgs(Args).needsAsanRt())
      LibSuffix.append("/asan");
  }

  // If device debugging turned on, add specially built bc files
  StringRef libpath = Args.MakeArgString(C.getDriver().Dir + "/../" + LibSuffix);
  std::string lib_debug_perf_path = FindDebugPerfInLibraryPath(LibSuffix);
  if (!lib_debug_perf_path.empty())
    libpath = lib_debug_perf_path;

  llvm::SmallVector<std::string, 12> BCLibs;

  if (Args.hasFlag(options::OPT_fgpu_sanitize, options::OPT_fno_gpu_sanitize,
                   true) &&
      TC.getSanitizerArgs(Args).needsAsanRt()) {
    std::string AsanRTL(RocmInstallation.getAsanRTLPath());
    if (AsanRTL.empty()) {
      if (!Args.hasArg(options::OPT_nogpulib))
        TC.getDriver().Diag(diag::err_drv_no_asan_rt_lib);
    } else {
      BCLibs.push_back(AsanRTL);
    }
  }
  StringRef GPUArch = getProcessorFromTargetID(Triple, TargetID);

  BCLibs.push_back(
      Args.MakeArgString(libpath + "/libomptarget-amdgpu-" + GPUArch + ".bc"));

  // Add the generic set of libraries, OpenMP subset only
  BCLibs.append(amdgpu::dlr::getCommonDeviceLibNames(
      C.getArgs(), C.getDriver(), GPUArch.str(), /* isOpenMP=*/true,
      RocmInstallation));

  llvm::for_each(BCLibs, [&](StringRef BCFile) {
    LastLinkArgs.push_back(Args.MakeArgString(BCFile));
  });

  LastLinkArgs.push_back("-o");
  auto LastLinkFileName =
      getOutputFileName(C, OutputFilePrefix, "-linked", "bc");
  LastLinkArgs.push_back(LastLinkFileName);

  return LastLinkFileName;
}

const char *amdgpu::dlr::getOptCommandArgs(Compilation &C,
                                           const llvm::opt::ArgList &Args,
                                           llvm::opt::ArgStringList &OptArgs,
                                           const llvm::Triple &Triple,
                                           llvm::StringRef TargetID,
                                           llvm::StringRef OutputFilePrefix,
                                           const char *InputFileName) {
  addAMDTargetArgs(C, Args, OptArgs, /*IsLlc*/ false);
  // OptArgs.push_back(Args.MakeArgString("-openmp-opt-disable=1"));

  OptArgs.push_back("-o");
  auto OutputFileName =
      getOutputFileName(C, OutputFilePrefix, "-optimized", "bc");
  OptArgs.push_back(OutputFileName);
  addCommonArgs(C, Args, OptArgs, Triple, TargetID, InputFileName,
                "ROCM_OPT_ARGS");

  return OutputFileName;
}

const char *amdgpu::dlr::getLlcCommandArgs(
    Compilation &C, const llvm::opt::ArgList &Args,
    llvm::opt::ArgStringList &LlcArgs, const llvm::Triple &Triple,
    llvm::StringRef TargetID, llvm::StringRef OutputFilePrefix,
    const char *InputFileName, bool OutputIsAsm) {
  addAMDTargetArgs(C, Args, LlcArgs, /*IsLLc*/ true);

  if (Arg *A = Args.getLastArgNoClaim(options::OPT_g_Group))
    if (!A->getOption().matches(options::OPT_g0) &&
        !A->getOption().matches(options::OPT_ggdb0))
      LlcArgs.push_back("-amdgpu-spill-cfi-saved-regs");

  LlcArgs.push_back(
      Args.MakeArgString(Twine("-filetype=") + (OutputIsAsm ? "asm" : "obj")));

  // Add output filename
  LlcArgs.push_back("-o");
  const char *LlcOutputFile =
      getOutputFileName(C, OutputFilePrefix, "", OutputIsAsm ? "s" : "o");
  LlcArgs.push_back(LlcOutputFile);
  addCommonArgs(C, Args, LlcArgs, Triple, TargetID, InputFileName,
                "ROCM_LLC_ARGS");

  return LlcOutputFile;
}

const char *amdgpu::dlr::getLldCommandArgs(
    Compilation &C, const InputInfo &Output, const llvm::opt::ArgList &Args,
    llvm::opt::ArgStringList &LldArgs, const llvm::Triple &Triple,
    llvm::StringRef TargetID, const char *InputFileName,
    const std::optional<std::string> OutputFilePrefix) {
  LldArgs.push_back("-flavor");
  LldArgs.push_back("gnu");
  LldArgs.push_back("--no-undefined");
  LldArgs.push_back("-shared");

  if (Arg *A = Args.getLastArgNoClaim(options::OPT_g_Group))
    if (!A->getOption().matches(options::OPT_g0) &&
        !A->getOption().matches(options::OPT_ggdb0))
      LldArgs.push_back("-plugin-opt=-amdgpu-spill-cfi-saved-regs");

  addCommonArgs(C, Args, LldArgs, Triple, TargetID, InputFileName,
                "ROCM_LLD_ARGS", /* isLld */ true);

  LldArgs.push_back("-o");
  const char *LldOutputFile =
      OutputFilePrefix ? getOutputFileName(C, *OutputFilePrefix, "", "out")
                       : Output.getFilename();
  LldArgs.push_back(LldOutputFile);

  return LldOutputFile;
}

AMDGPUOpenMPToolChain::AMDGPUOpenMPToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args,
                             const Action::OffloadKind OK)
    : ROCMToolChain(D, Triple, Args), HostTC(HostTC), OK(OK) {
  // Lookup binaries into the driver directory, this is used to
  // discover the 'amdgpu-arch' executable.
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

  if (DriverArgs.hasFlag(options::OPT_fgpu_approx_transcendentals,
                         options::OPT_fno_gpu_approx_transcendentals, false))
    CC1Args.push_back("-fcuda-approx-transcendentals");

  if (DriverArgs.hasFlag(options::OPT_fgpu_rdc, options::OPT_fno_gpu_rdc,
                         false))
    CC1Args.push_back("-fgpu-rdc");

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
       RocmInstallation->getRocmDeviceLibPathArg())
    LibraryPaths.push_back(DriverArgs.MakeArgString(Path));

  // Link the bitcode library late if we're using device LTO.
  if (getDriver().isUsingLTO(/* IsOffload */ true))
    return;

  std::string BitcodeSuffix;
  BitcodeSuffix = llvm::Twine("old-amdgpu-" + GPUArch).str();

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

  // Force APU mode will focefully include #pragma omp requires
  // unified_shared_memory via the force_usm header
  if (DriverArgs.hasArg(options::OPT_fopenmp_force_usm)) {
    CC1Args.push_back("-include");
    CC1Args.push_back(
        DriverArgs.MakeArgString(HostTC.getDriver().ResourceDir +
                                 "/include/openmp_wrappers/force_usm.h"));
  }
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

llvm::SmallVector<ToolChain::BitCodeLibraryInfo, 12>
AMDGPUOpenMPToolChain::getDeviceLibs(const llvm::opt::ArgList &Args) const {
  if (Args.hasArg(options::OPT_nogpulib))
    return {};

  if (!RocmInstallation->hasDeviceLibrary()) {
    getDriver().Diag(diag::err_drv_no_rocm_device_lib) << 0;
    return {};
  }

  StringRef GpuArch = getProcessorFromTargetID(
      getTriple(), Args.getLastArgValue(options::OPT_march_EQ));

  SmallVector<BitCodeLibraryInfo, 12> BCLibs;
  for (auto BCLib : getCommonDeviceLibNames(Args, GpuArch.str(),
                                            /*IsOpenMP=*/true))
    BCLibs.emplace_back(BCLib);

  return BCLibs;
}
