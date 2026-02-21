//===--- ModernizeTidyModule.cpp - flang-tidy -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../FlangTidyModule.h"
#include "../FlangTidyModuleRegistry.h"
#include "AvoidAssignStmt.h"
#include "AvoidBackspaceStmt.h"
#include "AvoidCommonBlocks.h"
#include "AvoidDataConstructs.h"
#include "AvoidPauseStmt.h"

namespace Fortran::tidy {
namespace modernize {

class ModernizeModule : public FlangTidyModule {
public:
  void addCheckFactories(FlangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<AvoidAssignStmtCheck>(
        "modernize-avoid-assign-stmt");
    CheckFactories.registerCheck<AvoidBackspaceStmtCheck>(
        "modernize-avoid-backspace-stmt");
    CheckFactories.registerCheck<AvoidCommonBlocksCheck>(
        "modernize-avoid-common-blocks");
    CheckFactories.registerCheck<AvoidDataConstructsCheck>(
        "modernize-avoid-data-constructs");
    CheckFactories.registerCheck<AvoidPauseStmtCheck>(
        "modernize-avoid-pause-stmt");
  }
};

} // namespace modernize

// Register the BugproneTidyModule using this statically initialized variable.
static FlangTidyModuleRegistry::Add<modernize::ModernizeModule>
    X("modernize-module", "Adds checks to enforce modern code style.");

// This anchor is used to force the linker to link in the generated object file
// and thus register the ModernizeModule.

// NOLINTNEXTLINE
volatile int ModernizeModuleAnchorSource = 0;

} // namespace Fortran::tidy
