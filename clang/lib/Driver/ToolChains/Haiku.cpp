//===--- Haiku.cpp - Haiku ToolChain Implementations ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Haiku.h"
#include "CommonArgs.h"
#include "clang/Config/config.h"
#include "llvm/Support/Path.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

/// Haiku - Haiku tool chain which can call as(1) and ld(1) directly.

Haiku::Haiku(const Driver &D, const llvm::Triple& Triple, const ArgList &Args)
  : Generic_ELF(D, Triple, Args) {

  getFilePaths().push_back(concat(getDriver().SysRoot, "/boot/system/lib"));
  getFilePaths().push_back(concat(getDriver().SysRoot, "/boot/system/develop/lib"));
}

void Haiku::AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
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

  // Add dirs specified via 'configure --with-c-include-dirs'.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (!CIncludeDirs.empty()) {
    SmallVector<StringRef, 5> dirs;
    CIncludeDirs.split(dirs, ":");
    for (StringRef dir : dirs) {
      StringRef Prefix =
        llvm::sys::path::is_absolute(dir) ? StringRef(D.SysRoot) : "";
      addExternCSystemInclude(DriverArgs, CC1Args, Prefix + dir);
    }
    return;
  }

  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/non-packaged/develop/headers"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/app"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/device"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/drivers"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/game"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/interface"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/kernel"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/locale"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/mail"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/media"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/midi"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/midi2"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/net"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/opengl"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/storage"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/support"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/translation"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/add-ons/graphics"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/add-ons/input_server"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/add-ons/mail_daemon"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/add-ons/registrar"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/add-ons/screen_saver"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/add-ons/tracker"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/be_apps/Deskbar"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/be_apps/NetPositive"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/os/be_apps/Tracker"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/3rdparty"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/bsd"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/glibc"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/gnu"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers/posix"));
  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot,
                   "/boot/system/develop/headers"));
}

void Haiku::addLibCxxIncludePaths(const llvm::opt::ArgList &DriverArgs,
                                  llvm::opt::ArgStringList &CC1Args) const {
  addSystemInclude(DriverArgs, CC1Args,
                   concat(getDriver().SysRoot, "/boot/system/develop/headers/c++/v1"));
}

void Haiku::addLibStdCxxIncludePaths(const llvm::opt::ArgList &DriverArgs,
                                     llvm::opt::ArgStringList &CC1Args) const {
  addLibStdCXXIncludePaths(concat(getDriver().SysRoot, "/boot/system/develop/headers/c++"),
                           getTriple().str(), "", DriverArgs, CC1Args);
}
