//===--- CrossWindowsToolChain.cpp - Cross Windows Tool Chain -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ToolChains.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;

CrossWindowsToolChain::CrossWindowsToolChain(const Driver &D,
                                             const llvm::Triple &T,
                                             const llvm::opt::ArgList &Args)
    : Generic_GCC(D, T, Args) {
  if (GetCXXStdlibType(Args) == ToolChain::CST_Libstdcxx) {
    const std::string &SysRoot = D.SysRoot;

    // libstdc++ resides in /usr/lib, but depends on libgcc which is placed in
    // /usr/lib/gcc.
    getFilePaths().push_back(SysRoot + "/usr/lib");
    getFilePaths().push_back(SysRoot + "/usr/lib/gcc");
  }
}

bool CrossWindowsToolChain::IsUnwindTablesDefault() const {
  // FIXME: all non-x86 targets need unwind tables, however, LLVM currently does
  // not know how to emit them.
  return getArch() == llvm::Triple::x86_64;
}

bool CrossWindowsToolChain::isPICDefault() const {
  return getArch() == llvm::Triple::x86_64;
}

bool CrossWindowsToolChain::isPIEDefault() const {
  return getArch() == llvm::Triple::x86_64;
}

bool CrossWindowsToolChain::isPICDefaultForced() const {
  return getArch() == llvm::Triple::x86_64;
}

void CrossWindowsToolChain::
AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                          llvm::opt::ArgStringList &CC1Args) const {
  const Driver &D = getDriver();
  const std::string &SysRoot = D.SysRoot;

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  addSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/local/include");
  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> ResourceDir(D.ResourceDir);
    llvm::sys::path::append(ResourceDir, "include");
    addSystemInclude(DriverArgs, CC1Args, ResourceDir);
  }
  for (const auto &P : DriverArgs.getAllArgValues(options::OPT_isystem_after))
    addSystemInclude(DriverArgs, CC1Args, P);
  addExternCSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/include");
}

void CrossWindowsToolChain::
AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                             llvm::opt::ArgStringList &CC1Args) const {
  const llvm::Triple &Triple = getTriple();
  const std::string &SysRoot = getDriver().SysRoot;

  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  switch (GetCXXStdlibType(DriverArgs)) {
  case ToolChain::CST_Libcxx:
    addSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/include/c++/v1");
    break;

  case ToolChain::CST_Libstdcxx:
    addSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/include/c++");
    addSystemInclude(DriverArgs, CC1Args,
                     SysRoot + "/usr/include/c++/" + Triple.str());
    addSystemInclude(DriverArgs, CC1Args,
                     SysRoot + "/usr/include/c++/backwards");
  }
}

void CrossWindowsToolChain::
AddCXXStdlibLibArgs(const llvm::opt::ArgList &DriverArgs,
                    llvm::opt::ArgStringList &CC1Args) const {
  switch (GetCXXStdlibType(DriverArgs)) {
  case ToolChain::CST_Libcxx:
    CC1Args.push_back("-lc++");
    break;
  case ToolChain::CST_Libstdcxx:
    CC1Args.push_back("-lstdc++");
    CC1Args.push_back("-lmingw32");
    CC1Args.push_back("-lmingwex");
    CC1Args.push_back("-lgcc");
    CC1Args.push_back("-lmoldname");
    CC1Args.push_back("-lmingw32");
    break;
  }
}

clang::SanitizerMask CrossWindowsToolChain::getSupportedSanitizers() const {
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  Res |= SanitizerKind::Address;
  return Res;
}

Tool *CrossWindowsToolChain::buildLinker() const {
  return new tools::CrossWindows::Linker(*this);
}

Tool *CrossWindowsToolChain::buildAssembler() const {
  return new tools::CrossWindows::Assembler(*this);
}
