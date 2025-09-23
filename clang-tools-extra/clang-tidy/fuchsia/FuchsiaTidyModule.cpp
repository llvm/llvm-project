//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "../google/UnnamedNamespaceInHeaderCheck.h"
#include "DefaultArgumentsCallsCheck.h"
#include "DefaultArgumentsDeclarationsCheck.h"
#include "MultipleInheritanceCheck.h"
#include "OverloadedOperatorCheck.h"
#include "StaticallyConstructedObjectsCheck.h"
#include "TrailingReturnCheck.h"
#include "VirtualInheritanceCheck.h"

using namespace clang::ast_matchers;

namespace clang::tidy {
namespace fuchsia {

/// This module is for Fuchsia-specific checks.
class FuchsiaModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<DefaultArgumentsCallsCheck>(
        "fuchsia-default-arguments-calls");
    CheckFactories.registerCheck<DefaultArgumentsDeclarationsCheck>(
        "fuchsia-default-arguments-declarations");
    CheckFactories.registerCheck<google::build::UnnamedNamespaceInHeaderCheck>(
        "fuchsia-header-anon-namespaces");
    CheckFactories.registerCheck<MultipleInheritanceCheck>(
        "fuchsia-multiple-inheritance");
    CheckFactories.registerCheck<OverloadedOperatorCheck>(
        "fuchsia-overloaded-operator");
    CheckFactories.registerCheck<StaticallyConstructedObjectsCheck>(
        "fuchsia-statically-constructed-objects");
    CheckFactories.registerCheck<TrailingReturnCheck>(
        "fuchsia-trailing-return");
    CheckFactories.registerCheck<VirtualInheritanceCheck>(
        "fuchsia-virtual-inheritance");
  }
};
// Register the FuchsiaTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<FuchsiaModule>
    X("fuchsia-module", "Adds Fuchsia platform checks.");
} // namespace fuchsia

// This anchor is used to force the linker to link in the generated object file
// and thus register the FuchsiaModule.
volatile int FuchsiaModuleAnchorSource = 0; // NOLINT(misc-use-internal-linkage)

} // namespace clang::tidy
