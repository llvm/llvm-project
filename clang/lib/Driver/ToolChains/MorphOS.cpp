//===--- MorphOS.cpp - MorphOS ToolChain Implementations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MorphOS.h"
#include "clang/Config/config.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/SanitizerArgs.h"
#include "clang/Options/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

void morphos::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                      const InputInfo &Output,
                                      const InputInfoList &Inputs,
                                      const ArgList &Args,
                                      const char *LinkingOutput) const {
  const auto &ToolChain = static_cast<const MorphOS &>(getToolChain());
  ArgStringList CmdArgs;

  claimNoWarnArgs(Args);

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString((ToolChain.GetProgramPath("as")));
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileCurCP(),
                                         Exec, CmdArgs, Inputs, Output));
}

void morphos::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  const auto &ToolChain = static_cast<const MorphOS &>(getToolChain());
  const Driver &D = ToolChain.getDriver();
  ArgStringList CmdArgs;
  // FIXME: Discover GCC instead of hard-coding the version.
  const std::string GCCLibPath = D.SysRoot + "/lib/gcc-lib/ppc-morphos/15.1.0";
  const bool NoIxemul = Args.hasArg(options::OPT_noixemul);

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  CmdArgs.push_back("--defsym");
  CmdArgs.push_back("__abox__=1");
  CmdArgs.push_back("-Qy");

  CmdArgs.push_back("-Bstatic");
  if (NoIxemul) {
    Args.ClaimAllArgs(options::OPT_noixemul);
    CmdArgs.push_back("--flavor=libnix");
  } else {
    CmdArgs.push_back("--flavor=ixemul");
  }

  assert((Output.isFilename() || Output.isNothing()) && "Invalid output.");
  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles,
                   options::OPT_r)) {
    if (NoIxemul) {
      CmdArgs.push_back(Args.MakeArgString(
          GCCLibPath + "/../../../../ppc-morphos/lib/libnix/crt0i.o"));
      CmdArgs.push_back(Args.MakeArgString(GCCLibPath + "/libnix/ecrti.o"));
      CmdArgs.push_back(Args.MakeArgString(GCCLibPath + "/libnix/crtbegin.o"));
    } else {
      CmdArgs.push_back(Args.MakeArgString(
          GCCLibPath + "/../../../../ppc-morphos/lib/crt0i.o"));
      CmdArgs.push_back(Args.MakeArgString(GCCLibPath + "/ecrti.o"));
      CmdArgs.push_back(Args.MakeArgString(GCCLibPath + "/crtbegin.o"));
    }
  }

  CmdArgs.push_back(Args.MakeArgString("-L" + GCCLibPath));
  CmdArgs.push_back(
      Args.MakeArgString("-L" + GCCLibPath + "/../../../../ppc-morphos/lib"));
  CmdArgs.push_back(Args.MakeArgString("-L" + D.SysRoot + "/lib"));

  Args.addAllArgs(CmdArgs, {options::OPT_L, options::OPT_T_Group,
                            options::OPT_s, options::OPT_t});
  ToolChain.AddFilePathLibArgs(Args, CmdArgs);

  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs, JA);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs,
                   options::OPT_r)) {
    if (D.CCCIsCXX()) {
      if (ToolChain.ShouldLinkCXXStdlib(Args))
        ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
      CmdArgs.push_back("-lm");
    }

    // Silence warnings when linking C code with a C++ '-stdlib' argument.
    Args.ClaimAllArgs(options::OPT_stdlib_EQ);

    CmdArgs.push_back("--start-group");
    CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lgcc");
    CmdArgs.push_back("-labox");
    CmdArgs.push_back("-laboxstubs");
    CmdArgs.push_back("-lsavl");
    CmdArgs.push_back("--end-group");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles,
                   options::OPT_r)) {
    if (NoIxemul) {
      CmdArgs.push_back(Args.MakeArgString(GCCLibPath + "/libnix/crtend.o"));
      CmdArgs.push_back(Args.MakeArgString(GCCLibPath + "/libnix/ecrtn.o"));
    } else {
      CmdArgs.push_back(Args.MakeArgString(GCCLibPath + "/crtend.o"));
      CmdArgs.push_back(Args.MakeArgString(GCCLibPath + "/ecrtn.o"));
    }
  }

  const char *Exec = Args.MakeArgString(ToolChain.GetLinkerPath());
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileCurCP(),
                                         Exec, CmdArgs, Inputs, Output));
}

/// MorphOS - MorphOS tool chain which can call as(1) and ld(1) directly.

MorphOS::MorphOS(const Driver &D, const llvm::Triple &Triple,
                 const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  if (!Args.hasArg(options::OPT_nostdlib)) {
    getFilePaths().push_back(concat(getDriver().SysRoot, "/ppc-morphos/lib"));
  }
}

Tool *MorphOS::buildAssembler() const {
  return new tools::morphos::Assembler(*this);
}

Tool *MorphOS::buildLinker() const { return new tools::morphos::Linker(*this); }

ToolChain::CXXStdlibType MorphOS::GetDefaultCXXStdlibType() const {
  return ToolChain::CST_Libstdcxx;
}

void MorphOS::AddClangSystemIncludeArgs(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {
  const Driver &D = getDriver();

  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> Dir(D.ResourceDir);
    llvm::sys::path::append(Dir, "include");
    addSystemInclude(DriverArgs, CC1Args, Dir.str());
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // Check for configure-time C include directories.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (CIncludeDirs != "") {
    SmallVector<StringRef, 5> dirs;
    CIncludeDirs.split(dirs, ":");
    for (StringRef dir : dirs) {
      StringRef Prefix =
          llvm::sys::path::is_absolute(dir) ? StringRef(D.SysRoot) : "";
      addExternCSystemInclude(DriverArgs, CC1Args, Prefix + dir);
    }
    return;
  }

  if (DriverArgs.hasArg(options::OPT_noixemul)) {
    addExternCSystemInclude(DriverArgs, CC1Args,
                            concat(D.SysRoot, "/includestd"));
  }
  addExternCSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot, "/include"));
  addExternCSystemInclude(DriverArgs, CC1Args,
                          concat(D.SysRoot, "/usr/include"));
  addExternCSystemInclude(DriverArgs, CC1Args,
                          concat(D.SysRoot, "/os-include"));
}

void MorphOS::addLibStdCxxIncludePaths(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {
  // FIXME: Discover GCC instead of hard-coding the version.
  addLibStdCXXIncludePaths(
      concat(getDriver().SysRoot,
             "/lib/gcc-lib/ppc-morphos/15.1.0/include/c++"),
      "", "", DriverArgs, CC1Args);
}
