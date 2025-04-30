//===--- Next32ELF.cpp - Next32ELF ToolChain Implementations ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Next32ELF.h"
#include "Arch/Next32.h"
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
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

void tools::Next32::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                         const InputInfo &Output,
                                         const InputInfoList &Inputs,
                                         const ArgList &Args,
                                         const char *LinkingOutput) const {
  const toolchains::Next32LLVMToolChain &ToolChain =
      static_cast<const toolchains::Next32LLVMToolChain &>(getToolChain());
  const Driver &D = ToolChain.getDriver();
  std::string Triple = ToolChain.getTripleString();

  ArgStringList CmdArgs;

  Args.AddAllArgs(CmdArgs, options::OPT_L);

  ToolChain.AddFilePathLibArgs(Args, CmdArgs);

  CmdArgs.push_back(Args.MakeArgString("-L" + D.Dir + "/../lib/" + Triple));

  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs, JA);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    auto lib =
        Args.hasArg(options::OPT_shared) ? "-lnext32rt_so" : "-lnext32rt";
    CmdArgs.push_back(lib);
  }

  if (ToolChain.ShouldLinkCXXStdlib(Args)) {
    ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
  }

  CmdArgs.push_back("-unresolved-symbols=ignore-all");

  if (Args.hasArg(options::OPT_static))
    CmdArgs.push_back("-static");
  else if (Args.hasArg(options::OPT_shared))
    CmdArgs.push_back("-shared");

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (Args.hasArg(options::OPT_v))
    CmdArgs.push_back("-v");

  CmdArgs.push_back("--discard-none");
  CmdArgs.push_back("-Helf_amd64");
  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                         Exec, CmdArgs, Inputs, Output));
}

/// Next32 Toolchain
Next32LLVMToolChain::Next32LLVMToolChain(const Driver &D,
                                         const llvm::Triple &Triple,
                                         const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {}

unsigned Next32LLVMToolChain::GetDefaultDwarfVersion() const { return 4; }

void Next32LLVMToolChain::AddClangSystemIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  const Driver &D = getDriver();

  if (DriverArgs.hasArg(clang::driver::options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nostdlibinc)) {
    addExternCSystemInclude(DriverArgs, CC1Args, D.Dir + "/../include");

    addSystemInclude(DriverArgs, CC1Args, "/usr/local/include");
    addExternCSystemInclude(DriverArgs, CC1Args,
                            "/usr/include/x86_64-linux-gnu/");
    addExternCSystemInclude(DriverArgs, CC1Args, "/usr/include");
    addExternCSystemInclude(DriverArgs, CC1Args, "/include");
  }

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> P(D.ResourceDir);
    llvm::sys::path::append(P, "include");
    addSystemInclude(DriverArgs, CC1Args, P);
  }
}

void Next32LLVMToolChain::addClangTargetOptions(const ArgList &DriverArgs,
                                                ArgStringList &CC1Args,
                                                Action::OffloadKind) const {
  const bool UseInitArrayDefault = true;
  if (!DriverArgs.hasFlag(options::OPT_fuse_init_array,
                          options::OPT_fno_use_init_array, UseInitArrayDefault))
    CC1Args.push_back("-fno-use-init-array");
}

Tool *Next32LLVMToolChain::buildLinker() const {
  return new tools::Next32::Linker(*this);
}

const char *Next32LLVMToolChain::getDefaultLinker() const { return "ld.lld"; }

void Next32LLVMToolChain::AddCXXStdlibLibArgs(
    const llvm::opt::ArgList &Args, llvm::opt::ArgStringList &CmdArgs) const {
  assert((GetCXXStdlibType(Args) == ToolChain::CST_Libcxx) &&
         "Only -lc++ (aka libcxx) is supported in this toolchain.");

  CmdArgs.push_back("-lc++");
  CmdArgs.push_back("-lc++abi");
  CmdArgs.push_back("-lunwind");
}

ToolChain::CXXStdlibType Next32LLVMToolChain::GetDefaultCXXStdlibType() const {
  return CST_Libcxx;
}
