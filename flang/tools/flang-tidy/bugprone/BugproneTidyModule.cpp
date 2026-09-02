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
#include "MissingDefaultCheck.h"
#include "PrecisionLossCheck.h"
#include "ShortCircuitCheck.h"
#include "SubprogramTrampolineCheck.h"
#include "UndeclaredProcCheck.h"
#include "UnusedIntentCheck.h"

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
    CheckFactories.registerCheck<MissingDefaultCheck>(
        "bugprone-missing-default-case");
    CheckFactories.registerCheck<PrecisionLossCheck>("bugprone-precision-loss");
    CheckFactories.registerCheck<ShortCircuitCheck>("bugprone-short-circuit");
    CheckFactories.registerCheck<SubprogramTrampolineCheck>(
        "bugprone-subprogram-trampoline");
    CheckFactories.registerCheck<UndeclaredProcCheck>(
        "bugprone-undeclared-procedure");
    CheckFactories.registerCheck<UnusedIntentCheck>("bugprone-unused-intent");
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
