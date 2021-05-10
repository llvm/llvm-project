//===-- NanoMips.cpp - NaneMips ToolChain Implementations -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NanoMips.h"
#include "Arch/Mips.h"
#include "CommonArgs.h"
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

NanoMips::NanoMips(const Driver &D, const llvm::Triple &Triple,
           const llvm::opt::ArgList &Args) : Generic_ELF(D, Triple, Args) {
  GCCInstallation.init(Triple, Args);
  Multilibs = GCCInstallation.getMultilibs();
  SelectedMultilib = GCCInstallation.getMultilib();
}


void NanoMips::AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                                         llvm::opt::ArgStringList &CC1Args) const
{
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  AddMultilibIncludeArgs(DriverArgs, CC1Args);

  // Add the obvious include dir from the GCC install.
  if (GCCInstallation.isValid()) {
    addSystemInclude(DriverArgs, CC1Args, GCCInstallation.getInstallPath() + "/include");
  }

}
