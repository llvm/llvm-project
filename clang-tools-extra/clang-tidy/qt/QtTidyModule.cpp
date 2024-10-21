//===-- QtTidyModule.cpp - clang-tidy ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "../modernize/UseIntegerSignComparisonCheck.h"

namespace clang::tidy {
namespace qt {

/// A module containing checks of the C++ Core Guidelines
class QtClangTidyModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<modernize::UseIntegerSignComparisonCheck>(
        "qt-integer-sign-comparison");
  }

  ClangTidyOptions getModuleOptions() override {
    ClangTidyOptions Options;
    ClangTidyOptions::OptionMap &Opts = Options.CheckOptions;

    Opts["qt-integer-sign-comparison."
         "IsQtApplication"] = "true";

    return Options;
  }
};

// Register the LLVMTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<QtClangTidyModule>
    X("qt-module", "Adds checks for the Qt framework Guidelines.");

} // namespace qt

// This anchor is used to force the linker to link in the generated object file
// and thus register the QtClangTidyModule.
volatile int QtClangTidyModuleAnchorSource = 0;

} // namespace clang::tidy
