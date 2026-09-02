//===--- PybindTidyModule.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "CallGuardInitCheck.h"

namespace clang::tidy {
namespace pybind {

class PybindModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<CallGuardInitCheck>("pybind-call-guard-init");
  }
};

// Register the PybindModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<PybindModule> X("pybind-module",
                                                    "Add pybind checks.");

} // namespace pybind

// This anchor is used to force the linker to link in the generated object file
// and thus register the PybindModule.
volatile int PybindModuleAnchorSource = 0; // NOLINT(misc-use-internal-linkage)

} // namespace clang::tidy
