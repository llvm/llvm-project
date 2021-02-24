//===-- BareMetal.cpp - Bare Metal ToolChain --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BareMetal.h"

#include "CommonArgs.h"
#include "InputInfo.h"
#include "Gnu.h"

#include "Arch/RISCV.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm::opt;
using namespace clang;
using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;

static Multilib makeMultilib(StringRef commonSuffix) {
  return Multilib(commonSuffix, commonSuffix, commonSuffix);
}

static bool findRISCVMultilibs(const Driver &D,
                               const llvm::Triple &TargetTriple,
                               const ArgList &Args, DetectedMultilibs &Result) {
  Multilib::flags_list Flags;
  StringRef Arch = riscv::getRISCVArch(Args, TargetTriple);
  StringRef Abi = tools::riscv::getRISCVABI(Args, TargetTriple);

  if (TargetTriple.getArch() == llvm::Triple::riscv64) {
    Multilib Imac = makeMultilib("").flag("+march=rv64imac").flag("+mabi=lp64");
    Multilib Imafdc = makeMultilib("/rv64imafdc/lp64d")
                          .flag("+march=rv64imafdc")
                          .flag("+mabi=lp64d");

    // Multilib reuse
    bool UseImafdc =
        (Arch == "rv64imafdc") || (Arch == "rv64gc"); // gc => imafdc

    addMultilibFlag((Arch == "rv64imac"), "march=rv64imac", Flags);
    addMultilibFlag(UseImafdc, "march=rv64imafdc", Flags);
    addMultilibFlag(Abi == "lp64", "mabi=lp64", Flags);
    addMultilibFlag(Abi == "lp64d", "mabi=lp64d", Flags);

    Result.Multilibs = MultilibSet().Either(Imac, Imafdc);
    return Result.Multilibs.select(Flags, Result.SelectedMultilib);
  }
  if (TargetTriple.getArch() == llvm::Triple::riscv32) {
    Multilib Imac =
        makeMultilib("").flag("+march=rv32imac").flag("+mabi=ilp32");
    Multilib I =
        makeMultilib("/rv32i/ilp32").flag("+march=rv32i").flag("+mabi=ilp32");
    Multilib Im =
        makeMultilib("/rv32im/ilp32").flag("+march=rv32im").flag("+mabi=ilp32");
    Multilib Iac = makeMultilib("/rv32iac/ilp32")
                       .flag("+march=rv32iac")
                       .flag("+mabi=ilp32");
    Multilib Imafc = makeMultilib("/rv32imafc/ilp32f")
                         .flag("+march=rv32imafc")
                         .flag("+mabi=ilp32f");

    // Multilib reuse
    bool UseI = (Arch == "rv32i") || (Arch == "rv32ic");    // ic => i
    bool UseIm = (Arch == "rv32im") || (Arch == "rv32imc"); // imc => im
    bool UseImafc = (Arch == "rv32imafc") || (Arch == "rv32imafdc") ||
                    (Arch == "rv32gc"); // imafdc,gc => imafc

    addMultilibFlag(UseI, "march=rv32i", Flags);
    addMultilibFlag(UseIm, "march=rv32im", Flags);
    addMultilibFlag((Arch == "rv32iac"), "march=rv32iac", Flags);
    addMultilibFlag((Arch == "rv32imac"), "march=rv32imac", Flags);
    addMultilibFlag(UseImafc, "march=rv32imafc", Flags);
    addMultilibFlag(Abi == "ilp32", "mabi=ilp32", Flags);
    addMultilibFlag(Abi == "ilp32f", "mabi=ilp32f", Flags);

    Result.Multilibs = MultilibSet().Either(I, Im, Iac, Imac, Imafc);
    return Result.Multilibs.select(Flags, Result.SelectedMultilib);
  }
  return false;
}

BareMetal::BareMetal(const Driver &D, const llvm::Triple &Triple,
                           const ArgList &Args)
    : ToolChain(D, Triple, Args) {
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);

  findMultilibs(D, Triple, Args);
  SmallString<128> SysRoot(computeSysRoot());
  if (!SysRoot.empty()) {
    llvm::sys::path::append(SysRoot, "lib");
    getFilePaths().push_back(std::string(SysRoot));
  }
}

/// Is the triple {arm,thumb}-none-none-{eabi,eabihf} ?
static bool isARMBareMetal(const llvm::Triple &Triple) {
  if (Triple.getArch() != llvm::Triple::arm &&
      Triple.getArch() != llvm::Triple::thumb)
    return false;

  if (Triple.getVendor() != llvm::Triple::UnknownVendor)
    return false;

  if (Triple.getOS() != llvm::Triple::UnknownOS)
    return false;

  if (Triple.getEnvironment() != llvm::Triple::EABI &&
      Triple.getEnvironment() != llvm::Triple::EABIHF)
    return false;

  return true;
}

static bool isRISCVBareMetal(const llvm::Triple &Triple) {
  if (Triple.getArch() != llvm::Triple::riscv32 &&
      Triple.getArch() != llvm::Triple::riscv64)
    return false;

  if (Triple.getVendor() != llvm::Triple::UnknownVendor)
    return false;

  if (Triple.getOS() != llvm::Triple::UnknownOS)
    return false;

  return Triple.getEnvironmentName() == "elf";
}

void BareMetal::findMultilibs(const Driver &D, const llvm::Triple &Triple,
                              const ArgList &Args) {
  DetectedMultilibs Result;
  if (isRISCVBareMetal(Triple)) {
    if (findRISCVMultilibs(D, Triple, Args, Result)) {
      SelectedMultilib = Result.SelectedMultilib;
      Multilibs = Result.Multilibs;
    }
  }
}

bool BareMetal::handlesTarget(const llvm::Triple &Triple) {
  return isARMBareMetal(Triple) || isRISCVBareMetal(Triple);
}

Tool *BareMetal::buildLinker() const {
  return new tools::baremetal::Linker(*this);
}

std::string BareMetal::getCompilerRTPath() const { return getRuntimesDir(); }

std::string BareMetal::getCompilerRTBasename(const llvm::opt::ArgList &,
                                             StringRef, FileType, bool) const {
  return ("libclang_rt.builtins-" + getTriple().getArchName() + ".a").str();
}

std::string BareMetal::getRuntimesDir() const {
  SmallString<128> Dir(getDriver().ResourceDir);
  llvm::sys::path::append(Dir, "lib", "baremetal");
  Dir += SelectedMultilib.gccSuffix();
  return std::string(Dir.str());
}

std::string BareMetal::computeSysRoot() const {
  if (!getDriver().SysRoot.empty())
    return getDriver().SysRoot + SelectedMultilib.osSuffix();

  SmallString<128> SysRootDir;
  llvm::sys::path::append(SysRootDir, getDriver().Dir, "../lib/clang-runtimes",
                          getDriver().getTargetTriple());

  SysRootDir += SelectedMultilib.osSuffix();
  return std::string(SysRootDir);
}

void BareMetal::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                          ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> Dir(getDriver().ResourceDir);
    llvm::sys::path::append(Dir, "include");
    addSystemInclude(DriverArgs, CC1Args, Dir.str());
  }

  if (!DriverArgs.hasArg(options::OPT_nostdlibinc)) {
    SmallString<128> Dir(computeSysRoot());
    if (!Dir.empty()) {
      llvm::sys::path::append(Dir, "include");
      addSystemInclude(DriverArgs, CC1Args, Dir.str());
    }
  }
}

void BareMetal::addClangTargetOptions(const ArgList &DriverArgs,
                                      ArgStringList &CC1Args,
                                      Action::OffloadKind) const {
  CC1Args.push_back("-nostdsysteminc");
}

void BareMetal::AddClangCXXStdlibIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  std::string SysRoot(computeSysRoot());
  if (SysRoot.empty())
    return;

  switch (GetCXXStdlibType(DriverArgs)) {
  case ToolChain::CST_Libcxx: {
    SmallString<128> Dir(SysRoot);
    llvm::sys::path::append(Dir, "include", "c++", "v1");
    addSystemInclude(DriverArgs, CC1Args, Dir.str());
    break;
  }
  case ToolChain::CST_Libstdcxx: {
    SmallString<128> Dir(SysRoot);
    llvm::sys::path::append(Dir, "include", "c++");
    std::error_code EC;
    Generic_GCC::GCCVersion Version = {"", -1, -1, -1, "", "", ""};
    // Walk the subdirs, and find the one with the newest gcc version:
    for (llvm::vfs::directory_iterator
             LI = getDriver().getVFS().dir_begin(Dir.str(), EC),
             LE;
         !EC && LI != LE; LI = LI.increment(EC)) {
      StringRef VersionText = llvm::sys::path::filename(LI->path());
      auto CandidateVersion = Generic_GCC::GCCVersion::Parse(VersionText);
      if (CandidateVersion.Major == -1)
        continue;
      if (CandidateVersion <= Version)
        continue;
      Version = CandidateVersion;
    }
    if (Version.Major == -1)
      return;
    llvm::sys::path::append(Dir, Version.Text);
    addSystemInclude(DriverArgs, CC1Args, Dir.str());
    break;
  }
  }
}

void BareMetal::AddCXXStdlibLibArgs(const ArgList &Args,
                                    ArgStringList &CmdArgs) const {
  switch (GetCXXStdlibType(Args)) {
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-lc++");
    CmdArgs.push_back("-lc++abi");
    break;
  case ToolChain::CST_Libstdcxx:
    CmdArgs.push_back("-lstdc++");
    CmdArgs.push_back("-lsupc++");
    break;
  }
  CmdArgs.push_back("-lunwind");
}

void BareMetal::AddLinkRuntimeLib(const ArgList &Args,
                                  ArgStringList &CmdArgs) const {
  ToolChain::RuntimeLibType RLT = GetRuntimeLibType(Args);
  switch (RLT) {
  case ToolChain::RLT_CompilerRT:
    CmdArgs.push_back(
        Args.MakeArgString("-lclang_rt.builtins-" + getTriple().getArchName()));
    return;
  case ToolChain::RLT_Libgcc:
    CmdArgs.push_back("-lgcc");
    return;
  }
  llvm_unreachable("Unhandled RuntimeLibType.");
}

void baremetal::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                     const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  auto &TC = static_cast<const toolchains::BareMetal&>(getToolChain());

  AddLinkerInputs(TC, Inputs, Args, CmdArgs, JA);

  CmdArgs.push_back("-Bstatic");

  CmdArgs.push_back(Args.MakeArgString("-L" + TC.getRuntimesDir()));

  TC.AddFilePathLibArgs(Args, CmdArgs);
  Args.AddAllArgs(CmdArgs, {options::OPT_L, options::OPT_T_Group,
                            options::OPT_e, options::OPT_s, options::OPT_t,
                            options::OPT_Z_Flag, options::OPT_r});

  if (TC.ShouldLinkCXXStdlib(Args))
    TC.AddCXXStdlibLibArgs(Args, CmdArgs);
  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lm");

    TC.AddLinkRuntimeLib(Args, CmdArgs);
  }

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  C.addCommand(std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                         Args.MakeArgString(TC.GetLinkerPath()),
                                         CmdArgs, Inputs, Output));
}
