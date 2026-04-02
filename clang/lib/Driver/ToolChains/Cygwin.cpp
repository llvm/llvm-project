//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Cygwin.h"
#include "clang/Config/config.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Driver.h"
#include "clang/Options/Options.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

using tools::addPathIfExists;

Cygwin::Cygwin(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : Generic_GCC(D, Triple, Args) {
  GCCInstallation.init(Triple, Args);
  std::string SysRoot = computeSysRoot();
  ToolChain::path_list &PPaths = getProgramPaths();

  Generic_GCC::PushPPaths(PPaths);

  path_list &Paths = getFilePaths();

  Generic_GCC::AddMultiarchPaths(D, SysRoot, "lib", Paths);

  // Similar to the logic for GCC above, if we are currently running Clang
  // inside of the requested system root, add its parent library path to those
  // searched.
  // FIXME: It's not clear whether we should use the driver's installed
  // directory ('Dir' below) or the ResourceDir.
  if (StringRef(D.Dir).starts_with(SysRoot))
    addPathIfExists(D, D.Dir + "/../lib", Paths);

  addPathIfExists(D, SysRoot + "/lib", Paths);
  addPathIfExists(D, SysRoot + "/usr/lib", Paths);
  addPathIfExists(D, SysRoot + "/usr/lib/w32api", Paths);
}

llvm::ExceptionHandling Cygwin::GetExceptionModel(const ArgList &Args) const {
  if (getArch() == llvm::Triple::x86_64 || getArch() == llvm::Triple::aarch64 ||
      getArch() == llvm::Triple::arm || getArch() == llvm::Triple::thumb)
    return llvm::ExceptionHandling::WinEH;
  return llvm::ExceptionHandling::DwarfCFI;
}

void Cygwin::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                       ArgStringList &CC1Args) const {
  const Driver &D = getDriver();
  std::string SysRoot = computeSysRoot();

  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nostdlibinc))
    addSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/local/include");

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> P(D.ResourceDir);
    llvm::sys::path::append(P, "include");
    addSystemInclude(DriverArgs, CC1Args, P);
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // Check for configure-time C include directories.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (CIncludeDirs != "") {
    SmallVector<StringRef, 5> Dirs;
    CIncludeDirs.split(Dirs, ":");
    for (StringRef Dir : Dirs) {
      StringRef Prefix =
          llvm::sys::path::is_absolute(Dir) ? "" : StringRef(SysRoot);
      addExternCSystemInclude(DriverArgs, CC1Args, Prefix + Dir);
    }
    return;
  }

  // Lacking those, try to detect the correct set of system includes for the
  // target triple.

  AddMultilibIncludeArgs(DriverArgs, CC1Args);

  // On systems using multiarch, add /usr/include/$triple before
  // /usr/include.
  std::string MultiarchIncludeDir = getTriple().str();
  if (!MultiarchIncludeDir.empty() &&
      D.getVFS().exists(SysRoot + "/usr/include/" + MultiarchIncludeDir))
    addExternCSystemInclude(DriverArgs, CC1Args,
                            SysRoot + "/usr/include/" + MultiarchIncludeDir);

  // Add an include of '/include' directly. This isn't provided by default by
  // system GCCs, but is often used with cross-compiling GCCs, and harmless to
  // add even when Clang is acting as-if it were a system compiler.
  addExternCSystemInclude(DriverArgs, CC1Args, SysRoot + "/include");

  addExternCSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/include");
  addExternCSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/include/w32api");
}
