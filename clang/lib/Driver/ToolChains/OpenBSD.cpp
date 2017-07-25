//===--- OpenBSD.cpp - OpenBSD ToolChain Implementations --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OpenBSD.h"
#include "Arch/Mips.h"
#include "Arch/Sparc.h"
#include "CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

void openbsd::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                      const InputInfo &Output,
                                      const InputInfoList &Inputs,
                                      const ArgList &Args,
                                      const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  switch (getToolChain().getArch()) {
  case llvm::Triple::x86:
    // When building 32-bit code on OpenBSD/amd64, we have to explicitly
    // instruct as in the base system to assemble 32-bit code.
    CmdArgs.push_back("--32");
    break;

  case llvm::Triple::ppc:
    CmdArgs.push_back("-mppc");
    CmdArgs.push_back("-many");
    break;

  case llvm::Triple::sparc:
  case llvm::Triple::sparcel: {
    CmdArgs.push_back("-32");
    std::string CPU = getCPUName(Args, getToolChain().getTriple());
    CmdArgs.push_back(sparc::getSparcAsmModeForCPU(CPU, getToolChain().getTriple()));
    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }

  case llvm::Triple::sparcv9: {
    CmdArgs.push_back("-64");
    std::string CPU = getCPUName(Args, getToolChain().getTriple());
    CmdArgs.push_back(sparc::getSparcAsmModeForCPU(CPU, getToolChain().getTriple()));
    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }

  case llvm::Triple::mips64:
  case llvm::Triple::mips64el: {
    StringRef CPUName;
    StringRef ABIName;
    mips::getMipsCPUAndABI(Args, getToolChain().getTriple(), CPUName, ABIName);

    CmdArgs.push_back("-mabi");
    CmdArgs.push_back(mips::getGnuCompatibleMipsABIName(ABIName).data());

    if (getToolChain().getArch() == llvm::Triple::mips64)
      CmdArgs.push_back("-EB");
    else
      CmdArgs.push_back("-EL");

    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }

  default:
    break;
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void openbsd::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  const Driver &D = getToolChain().getDriver();
  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);

  if (getToolChain().getArch() == llvm::Triple::mips64)
    CmdArgs.push_back("-EB");
  else if (getToolChain().getArch() == llvm::Triple::mips64el)
    CmdArgs.push_back("-EL");

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_shared)) {
    CmdArgs.push_back("-e");
    CmdArgs.push_back("__start");
  }

  CmdArgs.push_back("--eh-frame-hdr");
  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
  } else {
    if (Args.hasArg(options::OPT_rdynamic))
      CmdArgs.push_back("-export-dynamic");
    CmdArgs.push_back("-Bdynamic");
    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-shared");
    } else {
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back("/usr/libexec/ld.so");
    }
  }

  if (Args.hasArg(options::OPT_pie))
    CmdArgs.push_back("-pie");
  if (Args.hasArg(options::OPT_nopie))
    CmdArgs.push_back("-nopie");

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared)) {
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back(
            Args.MakeArgString(getToolChain().GetFilePath("gcrt0.o")));
      else if (Args.hasArg(options::OPT_static) &&
               !Args.hasArg(options::OPT_nopie))
        CmdArgs.push_back(
            Args.MakeArgString(getToolChain().GetFilePath("rcrt0.o")));
      else
        CmdArgs.push_back(
            Args.MakeArgString(getToolChain().GetFilePath("crt0.o")));
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtbegin.o")));
    } else {
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtbeginS.o")));
    }
  }

  std::string Triple = getToolChain().getTripleString();
  if (Triple.substr(0, 6) == "x86_64")
    Triple.replace(0, 6, "amd64");
  CmdArgs.push_back(
      Args.MakeArgString("-L/usr/lib/gcc-lib/" + Triple + "/4.2.1"));

  Args.AddAllArgs(CmdArgs, {options::OPT_L, options::OPT_T_Group,
                            options::OPT_e, options::OPT_s, options::OPT_t,
                            options::OPT_Z_Flag, options::OPT_r});

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs, JA);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    if (D.CCCIsCXX()) {
      if (getToolChain().ShouldLinkCXXStdlib(Args))
        getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lm_p");
      else
        CmdArgs.push_back("-lm");
    }

    // FIXME: For some reason GCC passes -lgcc before adding
    // the default system libraries. Just mimic this for now.
    CmdArgs.push_back("-lgcc");

    if (Args.hasArg(options::OPT_pthread)) {
      if (!Args.hasArg(options::OPT_shared) && Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lpthread_p");
      else
        CmdArgs.push_back("-lpthread");
    }

    if (!Args.hasArg(options::OPT_shared)) {
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lc_p");
      else
        CmdArgs.push_back("-lc");
    }

    CmdArgs.push_back("-lgcc");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
    else
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtendS.o")));
  }

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

/// OpenBSD - OpenBSD tool chain which can call as(1) and ld(1) directly.

OpenBSD::OpenBSD(const Driver &D, const llvm::Triple &Triple,
                 const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
}

Tool *OpenBSD::buildAssembler() const {
  return new tools::openbsd::Assembler(*this);
}

Tool *OpenBSD::buildLinker() const { return new tools::openbsd::Linker(*this); }
