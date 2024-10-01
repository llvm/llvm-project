//===------- BoostTidyModule.cpp - clang-tidy -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "UseRangesCheck.h"
#include "UseToStringCheck.h"
using namespace clang::ast_matchers;

namespace clang::tidy {
namespace boost {

class BoostModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<UseRangesCheck>("boost-use-ranges");
    CheckFactories.registerCheck<UseToStringCheck>("boost-use-to-string");
  }
};

// Register the BoostModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<BoostModule> X("boost-module",
                                                   "Add boost checks.");

} // namespace boost

// This anchor is used to force the linker to link in the generated object file
// and thus register the BoostModule.
volatile int BoostModuleAnchorSource = 0;

} // namespace clang::tidy
