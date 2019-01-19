//===--- LLVMTidyModule.cpp - clang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "../readability/NamespaceCommentCheck.h"
#include "HeaderGuardCheck.h"
#include "IncludeOrderCheck.h"
#include "TwineLocalCheck.h"

namespace clang {
namespace tidy {
namespace llvm {

class LLVMModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<LLVMHeaderGuardCheck>("llvm-header-guard");
    CheckFactories.registerCheck<IncludeOrderCheck>("llvm-include-order");
    CheckFactories.registerCheck<readability::NamespaceCommentCheck>(
        "llvm-namespace-comment");
    CheckFactories.registerCheck<TwineLocalCheck>("llvm-twine-local");
  }
};

// Register the LLVMTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<LLVMModule> X("llvm-module",
                                                  "Adds LLVM lint checks.");

} // namespace llvm

// This anchor is used to force the linker to link in the generated object file
// and thus register the LLVMModule.
volatile int LLVMModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
