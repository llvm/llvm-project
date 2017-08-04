//===--- Solaris.cpp - Solaris ToolChain Implementations --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Solaris.h"
#include "CommonArgs.h"
#include "clang/Config/config.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

void solaris::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                      const InputInfo &Output,
                                      const InputInfoList &Inputs,
                                      const ArgList &Args,
                                      const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void solaris::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  // Demangle C++ names in errors
  CmdArgs.push_back("-C");

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_shared)) {
    CmdArgs.push_back("-e");
    CmdArgs.push_back("_start");
  }

  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
    CmdArgs.push_back("-dn");
  } else {
    CmdArgs.push_back("-Bdynamic");
    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-shared");
    } else {
      CmdArgs.push_back("--dynamic-linker");
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("ld.so.1")));
    }
  }

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crt1.o")));

    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crti.o")));
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().GetFilePath("values-Xa.o")));
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().GetFilePath("crtbegin.o")));
  }

  getToolChain().AddFilePathLibArgs(Args, CmdArgs);

  Args.AddAllArgs(CmdArgs, {options::OPT_L, options::OPT_T_Group,
                            options::OPT_e, options::OPT_r});

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs, JA);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    if (getToolChain().getDriver().CCCIsCXX())
      getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);
    CmdArgs.push_back("-lgcc_s");
    CmdArgs.push_back("-lc");
    if (!Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-lgcc");
      CmdArgs.push_back("-lm");
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
  }
  CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crtn.o")));

  getToolChain().addProfileRTLibs(Args, CmdArgs);

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

/// Solaris - Solaris tool chain which can call as(1) and ld(1) directly.

Solaris::Solaris(const Driver &D, const llvm::Triple &Triple,
                 const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {

  GCCInstallation.init(Triple, Args);

  path_list &Paths = getFilePaths();
  if (GCCInstallation.isValid())
    addPathIfExists(D, GCCInstallation.getInstallPath(), Paths);

  addPathIfExists(D, getDriver().getInstalledDir(), Paths);
  if (getDriver().getInstalledDir() != getDriver().Dir)
    addPathIfExists(D, getDriver().Dir, Paths);

  addPathIfExists(D, getDriver().SysRoot + getDriver().Dir + "/../lib", Paths);

  std::string LibPath = "/usr/lib/";
  switch (Triple.getArch()) {
  case llvm::Triple::x86:
  case llvm::Triple::sparc:
    break;
  case llvm::Triple::x86_64:
    LibPath += "amd64/";
    break;
  case llvm::Triple::sparcv9:
    LibPath += "sparcv9/";
    break;
  default:
    llvm_unreachable("Unsupported architecture");
  }

  addPathIfExists(D, getDriver().SysRoot + LibPath, Paths);
}

Tool *Solaris::buildAssembler() const {
  return new tools::solaris::Assembler(*this);
}

Tool *Solaris::buildLinker() const { return new tools::solaris::Linker(*this); }

void Solaris::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                           ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  // Include the support directory for things like xlocale and fudged system
  // headers.
  // FIXME: This is a weird mix of libc++ and libstdc++. We should also be
  // checking the value of -stdlib= here and adding the includes for libc++
  // rather than libstdc++ if it's requested.
  addSystemInclude(DriverArgs, CC1Args, "/usr/include/c++/v1/support/solaris");

  if (GCCInstallation.isValid()) {
    GCCVersion Version = GCCInstallation.getVersion();
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/gcc/" +
                     Version.MajorStr + "." +
                     Version.MinorStr +
                     "/include/c++/" + Version.Text);
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/gcc/" + Version.MajorStr +
                     "." + Version.MinorStr + "/include/c++/" +
                     Version.Text + "/" +
                     GCCInstallation.getTriple().str());
  }
}
