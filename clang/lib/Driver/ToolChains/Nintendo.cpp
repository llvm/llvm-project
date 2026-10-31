//===--- Nintendo.cpp - Nintendo ToolChain Implementations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Nintendo.h"
#include "clang/Config/config.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/SanitizerArgs.h"
#include "clang/Options/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang::driver;
using namespace clang;
using namespace llvm::opt;

void tools::nintendo::Assembler::ConstructJob(Compilation &C,
                                              const JobAction &JA,
                                              const InputInfo &Output,
                                              const InputInfoList &Inputs,
                                              const ArgList &Args,
                                              const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  Args.AddLastArg(CmdArgs, options::OPT_march_EQ);
  Args.AddLastArg(CmdArgs, options::OPT_mcpu_EQ);
  Args.AddLastArg(CmdArgs, options::OPT_mfpu_EQ);

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Exec, CmdArgs, Inputs));

  // Handle the debug info splitting at object creation time if we're
  // creating an object.
  if (Args.hasArg(options::OPT_gsplit_dwarf))
    SplitDebugInfo(getToolChain(), C, *this, JA, Args, Output,
                   SplitDebugName(JA, Args, Inputs[0], Output));
}

void tools::nintendo::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                           const InputInfo &Output,
                                           const InputInfoList &Inputs,
                                           const ArgList &Args,
                                           const char *LinkingOutput) const {
  const toolchains::NintendoOS &ToolChain =
      static_cast<const toolchains::NintendoOS &>(getToolChain());
  const Driver &D = ToolChain.getDriver();
  const bool IsPIE =
      !Args.hasArg(options::OPT_shared) && !Args.hasArg(options::OPT_static) &&
      (Args.hasArg(options::OPT_pie) || ToolChain.isPIEDefault(Args));

  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);
  Args.ClaimAllArgs(options::OPT_nostartfiles);

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (IsPIE)
    CmdArgs.push_back("-pie");

  if (Args.hasArg(options::OPT_rdynamic))
    CmdArgs.push_back("-export-dynamic");

  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("-s");

  for (const auto &Opt : ToolChain.ExtraOpts)
    CmdArgs.push_back(Opt.c_str());

  if (!Args.hasArg(options::OPT_static))
    CmdArgs.push_back("--eh-frame-hdr");

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_u);

  ToolChain.AddFilePathLibArgs(Args, CmdArgs);

  if (auto LTO = ToolChain.getLTOMode(Args); LTO != LTOK_None)
    addLTOOptions(ToolChain, Args, CmdArgs, Output, Inputs, LTO == LTOK_Thin);

  if (Args.hasArg(options::OPT_Z_Xlinker__no_demangle))
    CmdArgs.push_back("--no-demangle");

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs, JA);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs,
                   options::OPT_r)) {
    addSanitizerRuntimes(ToolChain, Args, CmdArgs, C);

    // The profile runtime also needs access to system libraries.
    getToolChain().addProfileRTLibs(Args, CmdArgs);
  }

  const char *Exec = Args.MakeArgString(ToolChain.GetLinkerPath());
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Exec, CmdArgs, Inputs));
}

static bool validateToolChainRoot(const Driver &D, StringRef ToolChainRoot) {
  SmallString<128> ToolChainRootInclude = ToolChainRoot;
  llvm::sys::path::append(ToolChainRootInclude, "include");

  SmallString<128> ToolChainRootLib = ToolChainRoot;
  llvm::sys::path::append(ToolChainRootLib, "lib");

  if (!D.getVFS().exists(ToolChainRoot) ||
      !D.getVFS().exists(ToolChainRootInclude) ||
      !D.getVFS().exists(ToolChainRootLib))
    return false;

  return true;
}

static std::string getToolChainRoot(const Driver &D,
                                    const llvm::Triple &Triple) {
  // Use sysroot argument if it was provided.
  if (!D.SysRoot.empty())
    if (validateToolChainRoot(D, D.SysRoot))
      return D.SysRoot;

  // Use the toolchain root from the clang executable path otherwise.
  SmallString<128> ToolChainRoot(llvm::sys::path::parent_path(D.Dir));
  if (!ToolChainRoot.empty())
    if (validateToolChainRoot(D, ToolChainRoot))
      return std::string(ToolChainRoot);

  D.Diag(clang::diag::warn_drv_nintendo_bad_toolchain_root);
  return "";
}

toolchains::NintendoOS::NintendoOS(const Driver &D, const llvm::Triple &Triple,
                                   const ArgList &Args)
    : Generic_ELF(D, Triple, Args), ToolChainRoot(getToolChainRoot(D, Triple)) {

  path_list &ProgramPaths = getProgramPaths();

  SmallString<128> ToolChainBinPath(ToolChainRoot);
  llvm::sys::path::append(ToolChainBinPath, "bin");

  tools::addPathIfExists(D, ToolChainRoot, ProgramPaths);
  tools::addPathIfExists(D, ToolChainBinPath, ProgramPaths);
  tools::addPathIfExists(D, D.Dir, ProgramPaths);

  ExtraOpts.push_back("-z");
  ExtraOpts.push_back("relro");
  ExtraOpts.push_back("-X");
  ExtraOpts.push_back("--build-id=sha1");
  ExtraOpts.push_back("--hash-style=gnu");
  ExtraOpts.push_back("--pack-dyn-relocs=relr");
}

Tool *toolchains::NintendoOS::buildLinker() const {
  return new tools::nintendo::Linker(*this);
}

Tool *toolchains::NintendoOS::buildAssembler() const {
  return new tools::nintendo::Assembler(*this);
}

void toolchains::NintendoOS::AddClangSystemIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  if (!ToolChainRoot.empty()) {
    if (DriverArgs.hasArg(options::OPT_mkernel)) {
      SmallString<128> KernelIncPath(ToolChainRoot);
      llvm::sys::path::append(KernelIncPath, "include", "kernel");
      addSystemInclude(DriverArgs, CC1Args, KernelIncPath);
    }

    SmallString<128> IncPath(ToolChainRoot);
    SmallString<128> TargetIncPath(ToolChainRoot);
    llvm::sys::path::append(IncPath, "include");
    llvm::sys::path::append(TargetIncPath, "include", getTriple().str());
    addSystemInclude(DriverArgs, CC1Args, TargetIncPath);
    addSystemInclude(DriverArgs, CC1Args, IncPath);
  }

  // Check for configure-time C include directories.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (!CIncludeDirs.empty()) {
    SmallVector<StringRef, 5> dirs;
    CIncludeDirs.split(dirs, ":");
    for (StringRef dir : dirs)
      addExternCSystemInclude(DriverArgs, CC1Args, dir);
  }
}

ToolChain::CXXStdlibType
toolchains::NintendoOS::GetCXXStdlibType(const ArgList &Args) const {
  if (Arg *A = Args.getLastArg(options::OPT_stdlib_EQ)) {
    StringRef Value = A->getValue();
    if (Value == "libc++")
      return ToolChain::CST_Libcxx;
    if (Value == "platform")
      return ToolChain::CST_Libcxx;

    getDriver().Diag(diag::err_drv_invalid_stdlib_name) << A->getAsString(Args);
  }

  return ToolChain::CST_Libcxx;
}

ToolChain::RuntimeLibType
toolchains::NintendoOS::GetRuntimeLibType(const ArgList &Args) const {
  if (Arg *A = Args.getLastArg(options::OPT_rtlib_EQ)) {
    StringRef Value = A->getValue();
    if (Value != "compiler-rt")
      getDriver().Diag(diag::err_drv_invalid_rtlib_name)
          << A->getAsString(Args);
  }

  return ToolChain::RLT_CompilerRT;
}

void toolchains::NintendoOS::AddClangCXXStdlibIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc, options::OPT_nostdlibinc,
                        options::OPT_nostdincxx))
    return;

  if (ToolChainRoot.empty())
    return;

  // Claim -stdlib= options.
  (void)GetCXXStdlibType(DriverArgs);

  if (DriverArgs.hasArg(options::OPT_mkernel)) {
    SmallString<128> KernelIncPath(ToolChainRoot);
    llvm::sys::path::append(KernelIncPath, "include", "kernel", "c++", "v1");
    addSystemInclude(DriverArgs, CC1Args, KernelIncPath);
  } else {
    SmallString<128> IncPath(ToolChainRoot);
    SmallString<128> TargetIncPath(ToolChainRoot);
    llvm::sys::path::append(IncPath, "include", "c++", "v1");
    llvm::sys::path::append(TargetIncPath, "include", getTriple().str(), "c++",
                            "v1");
    addSystemInclude(DriverArgs, CC1Args, TargetIncPath);
    addSystemInclude(DriverArgs, CC1Args, IncPath);
  }
}

void toolchains::NintendoOS::addClangWarningOptions(
    ArgStringList &CC1Args) const {
  // Avoid default warning options.
}

bool toolchains::NintendoOS::isPIEDefault(const ArgList &Args) const {
  return getSanitizerArgs(Args).requiresPIE();
}

SanitizerMask toolchains::NintendoOS::getSupportedSanitizers(
    BoundArch BA, Action::OffloadKind DeviceOffloadKind) const {
  SanitizerMask Res = ToolChain::getSupportedSanitizers(BA, DeviceOffloadKind);
  Res |= SanitizerKind::Vptr;
  return Res;
}
