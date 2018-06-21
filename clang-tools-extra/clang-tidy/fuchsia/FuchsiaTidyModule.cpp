//===--- FuchsiaTidyModule.cpp - clang-tidy--------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "../google/UnnamedNamespaceInHeaderCheck.h"
#include "DefaultArgumentsCheck.h"
#include "MultipleInheritanceCheck.h"
#include "OverloadedOperatorCheck.h"
#include "RestrictSystemIncludesCheck.h"
#include "StaticallyConstructedObjectsCheck.h"
#include "TrailingReturnCheck.h"
#include "VirtualInheritanceCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace fuchsia {

/// This module is for Fuchsia-specific checks.
class FuchsiaModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<DefaultArgumentsCheck>(
        "fuchsia-default-arguments");
    CheckFactories.registerCheck<google::build::UnnamedNamespaceInHeaderCheck>(
        "fuchsia-header-anon-namespaces");
    CheckFactories.registerCheck<MultipleInheritanceCheck>(
        "fuchsia-multiple-inheritance");
    CheckFactories.registerCheck<OverloadedOperatorCheck>(
        "fuchsia-overloaded-operator");
    CheckFactories.registerCheck<RestrictSystemIncludesCheck>(
        "fuchsia-restrict-system-includes");
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
volatile int FuchsiaModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
