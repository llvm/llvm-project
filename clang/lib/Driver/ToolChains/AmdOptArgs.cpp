//===--- AmdOptArgs.cpp - Args handling for multiple toolchains -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Closed optimization compiler is invoked if -famd-opt is specified, or
// if any of the closed optimization flags are specified on the command line.
// These can also include -mllvm options as well as -f<options>
//
// Support is removed for -famd-opt, issue a warning.
//
//===----------------------------------------------------------------------===//

#include "CommonArgs.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

bool tools::checkForAMDProprietaryOptOptions(
    const ToolChain &TC, const Driver &D, const ArgList &Args,
    ArgStringList &CmdArgs, bool isLLD, bool checkOnly) {

  // -famd-opt enables prorietary compiler and lto
  if (Args.hasFlag(options::OPT_famd_opt, options::OPT_fno_amd_opt, false)) {
    D.Diag(diag::warn_drv_amd_opt_removed);
    return false;
  }
  // disables amd proprietary compiler
  if (Args.hasFlag(options::OPT_fno_amd_opt, options::OPT_famd_opt, false)) {
    return false;
  }
  return false;
}
