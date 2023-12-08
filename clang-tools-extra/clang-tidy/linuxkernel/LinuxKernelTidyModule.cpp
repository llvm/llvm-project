//===--- LinuxKernelTidyModule.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "MustCheckErrsCheck.h"

namespace clang::tidy {
namespace linuxkernel {

/// This module is for checks specific to the Linux kernel.
class LinuxKernelModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<MustCheckErrsCheck>(
        "linuxkernel-must-check-errs");
  }
};
// Register the LinuxKernelTidyModule using this statically initialized
// variable.
static ClangTidyModuleRegistry::Add<LinuxKernelModule>
    X("linux-module", "Adds checks specific to the Linux kernel.");
} // namespace linuxkernel

// This anchor is used to force the linker to link in the generated object file
// and thus register the LinuxKernelModule.
volatile int LinuxKernelModuleAnchorSource = 0;

} // namespace clang::tidy
