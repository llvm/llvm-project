//==- SPIRVOpenMP.cpp - SPIR-V OpenMP Tool Implementations --------*- C++ -*==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#include "SPIRVOpenMP.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Options/Options.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

namespace clang::driver::tools::SPIRVOpenMP {

std::string Linker::findSpirvTranslator(const llvm::opt::ArgList &Args) const {
  const ToolChain &TC = getToolChain();
  const std::string Major = std::to_string(LLVM_VERSION_MAJOR);

  // Prefer the AMD-patched translator that ships with ROCm toolchains.
  for (const std::string &Candidate :
       {"amd-llvm-spirv-" + Major, "amd-llvm-spirv",
        "llvm-spirv-" + Major, "llvm-spirv"}) {
    std::string Path = TC.GetProgramPath(Candidate.c_str());
    if (llvm::sys::fs::can_execute(Path))
      return Path;
  }
  // Return the last candidate regardless; the missing-executable error will
  // surface naturally when the command is executed.
  return TC.GetProgramPath("llvm-spirv");
}

void Linker::constructLinkAndEmitSpirvCommand(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const InputInfo &Output, const llvm::opt::ArgList &Args) const {

  assert(!Inputs.empty() && "Must have at least one input.");

  // Step 1: merge the compiled device bitcode inputs with llvm-link.
  // The device RTL (libomptarget-spirv.bc) was already absorbed into each
  // input at the cc1 stage via -mlink-builtin-bitcode; it must not be
  // re-linked here.
  const std::string TempBCPath = C.getDriver().GetTemporaryPath(
      llvm::sys::path::stem(Output.getFilename()), "bc");
  const char *TempBC = Args.MakeArgString(TempBCPath);

  ArgStringList LinkArgs;
  for (const InputInfo &Input : Inputs)
    LinkArgs.push_back(Input.getFilename());
  for (const Arg *A : Args.filtered(options::OPT_mlink_builtin_bitcode))
    LinkArgs.push_back(A->getValue());

  SmallString<128> LLVMLinkPath(C.getDriver().Dir);
  llvm::sys::path::append(LLVMLinkPath, "llvm-link");

  ArgStringList LLVMLinkCmdArgs = {"-o", TempBC};
  for (const char *Arg : LinkArgs)
    LLVMLinkCmdArgs.push_back(Arg);

  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::None(),
      Args.MakeArgString(LLVMLinkPath), LLVMLinkCmdArgs, Inputs,
      InputInfo(&JA, TempBC, TempBC)));

  // Step 2: translate the merged bitcode to SPIR-V.
  ArgStringList TrArgs;
  TrArgs.append({"--spirv-max-version=1.6", "--spirv-ext=+all",
                 "--spirv-allow-unknown-intrinsics",
                 "--spirv-lower-const-expr",
                 "--spirv-preserve-auxdata",
                 "--spirv-debug-info-version=nonsemantic-shader-200",
                 TempBC, "-o", Output.getFilename()});

  InputInfo TrInput(types::TY_LLVM_BC, TempBC, TempBC);
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::None(),
      Args.MakeArgString(findSpirvTranslator(Args)), TrArgs,
      TrInput, Output));
}

void Linker::ConstructJob(Compilation &C, const JobAction &JA,
                          const InputInfo &Output, const InputInfoList &Inputs,
                          const llvm::opt::ArgList &Args,
                          const char * /*LinkingOutput*/) const {
  constructLinkAndEmitSpirvCommand(C, JA, Inputs, Output, Args);
}

} // namespace clang::driver::tools::SPIRVOpenMP

namespace clang::driver::toolchains {

SPIRVOpenMPToolChain::SPIRVOpenMPToolChain(const Driver &D,
                                           const llvm::Triple &Triple,
                                           const ToolChain &HostToolchain,
                                           const ArgList &Args)
    : SPIRVToolChain(D, Triple, Args), HostTC(HostToolchain) {
  // Ensure the driver binary directory is searched first when locating
  // offload tools (e.g. llvm-link, amd-llvm-spirv).
  getProgramPaths().push_back(getDriver().Dir);
}

void SPIRVOpenMPToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
    BoundArch BA, Action::OffloadKind DeviceOffloadingKind) const {

  // Forward options the host toolchain has already collected (e.g. target
  // feature flags that must be visible to the device compilation).
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);

  if (DeviceOffloadingKind != Action::OFK_OpenMP)
    return;

  // Auto-vectorisation passes produce LLVM IR patterns that the SPIR-V
  // translator cannot handle reliably; disable them for device code.
  CC1Args.append({"-mllvm", "-vectorize-loops=false",
                  "-mllvm", "-vectorize-slp=false"});

  // Default to hidden visibility so that device symbols do not accidentally
  // leak into the host link unless explicitly exported.
  if (!DriverArgs.hasArg(options::OPT_fvisibility_EQ,
                         options::OPT_fvisibility_ms_compat))
    CC1Args.append({"-fvisibility=hidden",
                    "-fapply-global-visibility-to-externs"});

  // Inject the device RTL at compile time via -mlink-builtin-bitcode so that
  // it is absorbed during LTO codegen.  This follows the same pattern used by
  // HIPSPVToolChain and AMDGPUOpenMPToolChain and avoids having to re-link the
  // library in the separate llvm-link step.
  for (const BitCodeLibraryInfo &BCLib :
       getDeviceLibs(DriverArgs, Action::OFK_OpenMP))
    CC1Args.append({"-mlink-builtin-bitcode",
                    DriverArgs.MakeArgString(BCLib.Path)});
}

void SPIRVOpenMPToolChain::addClangWarningOptions(
    ArgStringList &CC1Args) const {
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
SPIRVOpenMPToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void SPIRVOpenMPToolChain::AddClangSystemIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void SPIRVOpenMPToolChain::AddClangCXXStdlibIncludeArgs(
    const ArgList &Args, ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}

llvm::SmallVector<ToolChain::BitCodeLibraryInfo, 12>
SPIRVOpenMPToolChain::getDeviceLibs(const llvm::opt::ArgList &DriverArgs,
                                    Action::OffloadKind) const {
  if (!DriverArgs.hasFlag(options::OPT_offloadlib, options::OPT_no_offloadlib,
                          true))
    return {};

  // Build the ordered list of directories to search for the device RTL.
  SmallVector<SmallString<128>, 6> SearchPaths;

  // 1. Explicit user override takes highest priority.
  if (StringRef UserPath = DriverArgs.getLastArgValue(
          options::OPT_libomptarget_spirv_bc_path_EQ);
      !UserPath.empty())
    SearchPaths.emplace_back(UserPath);

  // 2. ROCm installation lib directory.
  if (StringRef RocmPath =
          DriverArgs.getLastArgValue(options::OPT_rocm_path_EQ);
      !RocmPath.empty()) {
    SearchPaths.emplace_back(RocmPath);
    llvm::sys::path::append(SearchPaths.back(), "lib");
  }

  // 3. Sysroot lib directory.
  if (!getDriver().SysRoot.empty()) {
    SearchPaths.emplace_back(getDriver().SysRoot);
    llvm::sys::path::append(SearchPaths.back(), "lib");
  }

  // 4. Clang resource lib directory.
  SearchPaths.emplace_back(getDriver().ResourceDir);
  llvm::sys::path::append(SearchPaths.back(), "lib");

  // 5. Driver-relative lib directory (e.g. <install>/bin/../lib).
  SearchPaths.emplace_back(getDriver().Dir);
  llvm::sys::path::append(SearchPaths.back(), "..", "lib");

  constexpr llvm::StringLiteral BCName = "libomptarget-spirv.bc";
  for (const auto &Dir : SearchPaths) {
    SmallString<128> FullPath(Dir);
    llvm::sys::path::append(FullPath, BCName);
    if (llvm::sys::fs::exists(FullPath))
      return {{std::string(FullPath)}};
  }

  getDriver().Diag(diag::err_drv_omp_offload_target_missingbcruntime)
      << BCName << "spirv";
  return {};
}

SanitizerMask SPIRVOpenMPToolChain::getSupportedSanitizers() const {
  return HostTC.getSupportedSanitizers();
}

VersionTuple
SPIRVOpenMPToolChain::computeMSVCVersion(const Driver *D,
                                         const ArgList &Args) const {
  return HostTC.computeMSVCVersion(D, Args);
}

void SPIRVOpenMPToolChain::adjustDebugInfoKind(
    llvm::codegenoptions::DebugInfoKind &DebugInfoKind,
    const llvm::opt::ArgList &) const {
  // The SPIR-V translator currently aborts on DW_OP_LLVM_convert; suppress
  // debug info until the SPIR-V backend has full debug support.
  DebugInfoKind = llvm::codegenoptions::NoDebugInfo;
}

Tool *SPIRVOpenMPToolChain::buildLinker() const {
  return new tools::SPIRVOpenMP::Linker(*this);
}

} // namespace clang::driver::toolchains
