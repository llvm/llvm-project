//===--- WebAssembly.cpp - WebAssembly ToolChain Implementation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

wasm::Linker::Linker(const ToolChain &TC)
  : GnuTool("wasm::Linker", "lld", TC) {}

bool wasm::Linker::isLinkJob() const {
  return true;
}

bool wasm::Linker::hasIntegratedCPP() const {
  return false;
}

void wasm::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                const InputInfo &Output,
                                const InputInfoList &Inputs,
                                const ArgList &Args,
                                const char *LinkingOutput) const {

  const ToolChain &ToolChain = getToolChain();
  const char *Linker = Args.MakeArgString(ToolChain.GetLinkerPath());
  ArgStringList CmdArgs;
  CmdArgs.push_back("-flavor");
  CmdArgs.push_back("wasm");

  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("--strip-all");

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_u);
  ToolChain.AddFilePathLibArgs(Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles))
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crt1.o")));

  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs, JA);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    if (ToolChain.ShouldLinkCXXStdlib(Args))
      ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);

    if (Args.hasArg(options::OPT_pthread))
      CmdArgs.push_back("-lpthread");

    CmdArgs.push_back("-lc");
    AddRunTimeLibs(ToolChain, ToolChain.getDriver(), CmdArgs, Args);
  }

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  C.addCommand(llvm::make_unique<Command>(JA, *this, Linker, CmdArgs, Inputs));
}

WebAssembly::WebAssembly(const Driver &D, const llvm::Triple &Triple,
                         const llvm::opt::ArgList &Args)
  : ToolChain(D, Triple, Args) {

  assert(Triple.isArch32Bit() != Triple.isArch64Bit());

  getProgramPaths().push_back(getDriver().getInstalledDir());

  getFilePaths().push_back(getDriver().SysRoot + "/lib");
}

bool WebAssembly::IsMathErrnoDefault() const { return false; }

bool WebAssembly::IsObjCNonFragileABIDefault() const { return true; }

bool WebAssembly::UseObjCMixedDispatch() const { return true; }

bool WebAssembly::isPICDefault() const { return false; }

bool WebAssembly::isPIEDefault() const { return false; }

bool WebAssembly::isPICDefaultForced() const { return false; }

bool WebAssembly::IsIntegratedAssemblerDefault() const { return true; }

bool WebAssembly::hasBlocksRuntime() const { return false; }

// TODO: Support profiling.
bool WebAssembly::SupportsProfiling() const { return false; }

bool WebAssembly::HasNativeLLVMSupport() const { return true; }

void WebAssembly::addClangTargetOptions(const ArgList &DriverArgs,
                                        ArgStringList &CC1Args,
                                        Action::OffloadKind) const {
  if (DriverArgs.hasFlag(clang::driver::options::OPT_fuse_init_array,
                         options::OPT_fno_use_init_array, true))
    CC1Args.push_back("-fuse-init-array");
}

ToolChain::RuntimeLibType WebAssembly::GetDefaultRuntimeLibType() const {
  return ToolChain::RLT_CompilerRT;
}

ToolChain::CXXStdlibType WebAssembly::GetCXXStdlibType(const ArgList &Args) const {
  return ToolChain::CST_Libcxx;
}

void WebAssembly::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                            ArgStringList &CC1Args) const {
  if (!DriverArgs.hasArg(options::OPT_nostdinc))
    addSystemInclude(DriverArgs, CC1Args, getDriver().SysRoot + "/include");
}

void WebAssembly::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                               ArgStringList &CC1Args) const {
  if (!DriverArgs.hasArg(options::OPT_nostdlibinc) &&
      !DriverArgs.hasArg(options::OPT_nostdincxx))
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/include/c++/v1");
}

std::string WebAssembly::getThreadModel() const {
  // The WebAssembly MVP does not yet support threads; for now, use the
  // "single" threading model, which lowers atomics to non-atomic operations.
  // When threading support is standardized and implemented in popular engines,
  // this override should be removed.
  return "single";
}

Tool *WebAssembly::buildLinker() const {
  return new tools::wasm::Linker(*this);
}
