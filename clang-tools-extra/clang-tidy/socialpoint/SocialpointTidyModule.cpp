//===--- GoogleTidyModule.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "DefinitionsInHeadersCheck.h"
#include "SortConstructorInitializersCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace socialpoint {

class SocialpointModule : public ClangTidyModule {
 public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<DefinitionsInHeadersCheck>(
        "socialpoint-definitions-in-headers");
    CheckFactories.registerCheck<SortConstructorInitializersCheck>(
        "socialpoint-sort-constructor-initializers");
  }

  ClangTidyOptions getModuleOptions() override {
    ClangTidyOptions Options;
    auto &Opts = Options.CheckOptions;
    Opts["socialpoint-definitions-in-headers.IncludeInternalLinkage"] =
        "1";
    return Options;
  }
};

// Register the GoogleTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<SocialpointModule> X("socialpoint-module",
                                                         "Adds Socialpoint lint checks.");

}  // namespace socialpoint

// This anchor is used to force the linker to link in the generated object file
// and thus register the SocialpointModule.
volatile int SocialpointModuleAnchorSource = 0;

}  // namespace tidy
}  // namespace clang
