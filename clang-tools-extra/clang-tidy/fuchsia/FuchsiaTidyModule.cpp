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
#include "DefaultArgumentsCheck.h"
#include "OverloadedOperatorCheck.h"
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
    CheckFactories.registerCheck<OverloadedOperatorCheck>(
        "fuchsia-overloaded-operator");
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
