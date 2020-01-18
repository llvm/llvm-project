//===--- Xtensa.cpp - Xtensa ToolChain Implementations ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Xtensa.h"
#include "CommonArgs.h"
#include "InputInfo.h"
#include "clang/Basic/Cuda.h"
#include "clang/Config/config.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Distro.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <system_error>

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

XtensaGCCToolchainDetector::XtensaGCCToolchainDetector(
    const Driver &D, const llvm::Triple &HostTriple,
    const llvm::opt::ArgList &Args) {
  std::string InstalledDir;
  InstalledDir = D.getInstalledDir();
  StringRef CPUName = XtensaToolChain::GetTargetCPUVersion(Args);
  std::string Dir;
  std::string ToolchainName;
  std::string ToolchainDir;

  if (CPUName.equals("esp32"))
    ToolchainName = "xtensa-esp32-elf";
  else if (CPUName.equals("esp8266"))
    ToolchainName = "xtensa-lx106-elf";

  // ToolchainDir = InstalledDir + "/../" + ToolchainName;
  ToolchainDir = InstalledDir + "/..";
  Dir = ToolchainDir + "/lib/gcc/" + ToolchainName + "/";
  GCCLibAndIncVersion = "";

  if (D.getVFS().exists(Dir)) {
    std::error_code EC;
    for (llvm::vfs::directory_iterator LI = D.getVFS().dir_begin(Dir, EC), LE;
         !EC && LI != LE; LI = LI.increment(EC)) {
      StringRef VersionText = llvm::sys::path::filename(LI->path());
      auto GCCVersion = Generic_GCC::GCCVersion::Parse(VersionText);
      if (GCCVersion.Major == -1)
        continue;
      GCCLibAndIncVersion = GCCVersion.Text;
    }
    if (GCCLibAndIncVersion == "")
      llvm_unreachable("Unexpected Xtensa GCC toolchain version");

  } else {
    // Unable to find Xtensa GCC toolchain;
    GCCToolchainName = "";
    return;
  }
  GCCToolchainDir = ToolchainDir;
  GCCToolchainName = ToolchainName;
}

/// Xtensa Toolchain
XtensaToolChain::XtensaToolChain(const Driver &D, const llvm::Triple &Triple,
                                 const ArgList &Args)
    : Generic_ELF(D, Triple, Args), XtensaGCCToolchain(D, getTriple(), Args) {
  for (auto *A : Args) {
    std::string Str = A->getAsString(Args);
    if (!Str.compare("-mlongcalls"))
      A->claim();
    if (!Str.compare("-fno-tree-switch-conversion"))
      A->claim();

    // Currently don't use integrated assembler for assembler input files
    if ((IsIntegratedAsm) && (Str.length() > 2)) {
      std::string ExtSubStr = Str.substr(Str.length() - 2);
      if (!ExtSubStr.compare(".s"))
        IsIntegratedAsm = false;
      if (!ExtSubStr.compare(".S"))
        IsIntegratedAsm = false;
    }
  }

  // Currently don't use integrated assembler for assembler input files
  if (IsIntegratedAsm) {
    if (Args.getLastArgValue(options::OPT_x).equals("assembler"))
      IsIntegratedAsm = false;

    if (Args.getLastArgValue(options::OPT_x).equals("assembler-with-cpp"))
      IsIntegratedAsm = false;
  }
}

Tool *XtensaToolChain::buildLinker() const {
  return new tools::Xtensa::Linker(*this);
}

Tool *XtensaToolChain::buildAssembler() const {
  return new tools::Xtensa::Assembler(*this);
}

void XtensaToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                                ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(clang::driver::options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  if (!XtensaGCCToolchain.IsValid())
    return;

  std::string Path1 =
      XtensaGCCToolchain.GCCToolchainDir + "/lib/clang/10.0.0/include";
  std::string Path2 = XtensaGCCToolchain.GCCToolchainDir + "/lib/gcc/" +
                      XtensaGCCToolchain.GCCToolchainName + "/" +
                      XtensaGCCToolchain.GCCLibAndIncVersion + "/include";
  std::string Path3 = XtensaGCCToolchain.GCCToolchainDir + "/lib/gcc/" +
                      XtensaGCCToolchain.GCCToolchainName + "/" +
                      XtensaGCCToolchain.GCCLibAndIncVersion + "/include-fixed";
  std::string Path4 = XtensaGCCToolchain.GCCToolchainDir + "/" +
                      XtensaGCCToolchain.GCCToolchainName + "/sys-include";
  std::string Path5 = XtensaGCCToolchain.GCCToolchainDir + "/" +
                      XtensaGCCToolchain.GCCToolchainName + "/include";
  const StringRef Paths[] = {Path1, Path2, Path3, Path4, Path5};
  addSystemIncludes(DriverArgs, CC1Args, Paths);
}

void XtensaToolChain::addLibStdCxxIncludePaths(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {
  if (!XtensaGCCToolchain.IsValid())
    return;

  std::string BaseDir = XtensaGCCToolchain.GCCToolchainDir + "/" +
                        XtensaGCCToolchain.GCCToolchainName + "/include/c++/" +
                        XtensaGCCToolchain.GCCLibAndIncVersion;
  std::string TargetDir = BaseDir + "/" + XtensaGCCToolchain.GCCToolchainName;
  addLibStdCXXIncludePaths(BaseDir, "", "", "", "", "", DriverArgs, CC1Args);
  addLibStdCXXIncludePaths(TargetDir, "", "", "", "", "", DriverArgs, CC1Args);
  TargetDir = BaseDir + "/backward";
  addLibStdCXXIncludePaths(TargetDir, "", "", "", "", "", DriverArgs, CC1Args);
}

ToolChain::CXXStdlibType
XtensaToolChain::GetCXXStdlibType(const ArgList &Args) const {
  Arg *A = Args.getLastArg(options::OPT_stdlib_EQ);
  if (!A)
    return ToolChain::CST_Libstdcxx;

  StringRef Value = A->getValue();
  if (Value != "libstdc++")
    getDriver().Diag(diag::err_drv_invalid_stdlib_name) << A->getAsString(Args);

  return ToolChain::CST_Libstdcxx;
}

const StringRef XtensaToolChain::GetTargetCPUVersion(const ArgList &Args) {
  if (Arg *A = Args.getLastArg(clang::driver::options::OPT_mcpu_EQ)) {
    StringRef CPUName = A->getValue();
    return CPUName;
  }
  return "esp32";
}

void tools::Xtensa::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                            const InputInfo &Output,
                                            const InputInfoList &Inputs,
                                            const ArgList &Args,
                                            const char *LinkingOutput) const {
  const auto &TC =
      static_cast<const toolchains::XtensaToolChain &>(getToolChain());

  if (!TC.XtensaGCCToolchain.IsValid())
    llvm_unreachable("Unable to find Xtensa GCC assembler");

  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  CmdArgs.push_back("-c");

  if (Args.hasArg(options::OPT_v))
    CmdArgs.push_back("-v");

  if (Arg *A = Args.getLastArg(options::OPT_g_Group))
    if (!A->getOption().matches(options::OPT_g0))
      CmdArgs.push_back("-g");

  if (Args.hasFlag(options::OPT_fverbose_asm, options::OPT_fno_verbose_asm,
                   false))
    CmdArgs.push_back("-fverbose-asm");

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Asm =
      Args.MakeArgString(getToolChain().getDriver().Dir + "/" +
                         TC.XtensaGCCToolchain.GCCToolchainName + "-as");
  C.addCommand(llvm::make_unique<Command>(JA, *this, Asm, CmdArgs, Inputs));
}

void Xtensa::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {
  const auto &TC =
      static_cast<const toolchains::XtensaToolChain &>(getToolChain());

  if (!TC.XtensaGCCToolchain.IsValid())
    llvm_unreachable("Unable to find Xtensa GCC linker");

  std::string Linker = getToolChain().getDriver().Dir + "/" +
                       TC.XtensaGCCToolchain.GCCToolchainName + "-ld";
  ArgStringList CmdArgs;

  Args.AddAllArgs(CmdArgs,
                  {options::OPT_T_Group, options::OPT_e, options::OPT_s,
                   options::OPT_L, options::OPT_t, options::OPT_u_Group});

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs, JA);

  std::string Libs = TC.XtensaGCCToolchain.GCCToolchainDir + "/lib/gcc/" +
                     TC.XtensaGCCToolchain.GCCToolchainName + "/" +
                     TC.XtensaGCCToolchain.GCCLibAndIncVersion + "/";
  CmdArgs.push_back("-L");
  CmdArgs.push_back(Args.MakeArgString(Libs));

  Libs = TC.XtensaGCCToolchain.GCCToolchainDir + "/" +
         TC.XtensaGCCToolchain.GCCToolchainName + "/lib/";
  CmdArgs.push_back("-L");
  CmdArgs.push_back(Args.MakeArgString(Libs));

  CmdArgs.push_back("-v");

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Args.MakeArgString(Linker),
                                         CmdArgs, Inputs));
}
