//===--- BugproneTidyModule.cpp - flang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../FlangTidyModule.h"
#include "../FlangTidyModuleRegistry.h"
#include "ArithmeticGotoCheck.h"
#include "ArithmeticIfStmtCheck.h"
#include "ContiguousArrayCheck.h"
#include "ImplicitDeclCheck.h"
#include "ImpliedSaveCheck.h"
#include "MismatchedIntentCheck.h"
#include "MissingActionCheck.h"

namespace Fortran::tidy {
namespace bugprone {

class BugproneModule : public FlangTidyModule {
public:
  void addCheckFactories(FlangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<ArithmeticGotoCheck>(
        "bugprone-arithmetic-goto");
    CheckFactories.registerCheck<ArithmeticIfStmtCheck>(
        "bugprone-arithmetic-if");
    CheckFactories.registerCheck<ContiguousArrayCheck>(
        "bugprone-contiguous-array");
    CheckFactories.registerCheck<ImplicitDeclCheck>(
        "bugprone-implicit-declaration");
    CheckFactories.registerCheck<ImpliedSaveCheck>("bugprone-implied-save");
    CheckFactories.registerCheck<MismatchedIntentCheck>(
        "bugprone-mismatched-intent");
    CheckFactories.registerCheck<MissingActionCheck>("bugprone-missing-action");
  }
};

} // namespace bugprone

// Register the BugproneTidyModule using this statically initialized variable.
static FlangTidyModuleRegistry::Add<bugprone::BugproneModule>
    X("bugprone-module", "Adds checks for bugprone code constructs.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the BugproneModule.

// NOLINTNEXTLINE
volatile int BugproneModuleAnchorSource = 0;

} // namespace Fortran::tidy
