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
#include "../readability/ElseAfterReturnCheck.h"
#include "../readability/NamespaceCommentCheck.h"
#include "../readability/QualifiedAutoCheck.h"
#include "HeaderGuardCheck.h"
#include "IncludeOrderCheck.h"
#include "PreferIsaOrDynCastInConditionalsCheck.h"
#include "PreferRegisterOverUnsignedCheck.h"
#include "TwineLocalCheck.h"

namespace clang {
namespace tidy {
namespace llvm_check {

class LLVMModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<readability::ElseAfterReturnCheck>(
        "llvm-else-after-return");
    CheckFactories.registerCheck<LLVMHeaderGuardCheck>("llvm-header-guard");
    CheckFactories.registerCheck<IncludeOrderCheck>("llvm-include-order");
    CheckFactories.registerCheck<readability::NamespaceCommentCheck>(
        "llvm-namespace-comment");
    CheckFactories.registerCheck<PreferIsaOrDynCastInConditionalsCheck>(
        "llvm-prefer-isa-or-dyn-cast-in-conditionals");
    CheckFactories.registerCheck<PreferRegisterOverUnsignedCheck>(
        "llvm-prefer-register-over-unsigned");
    CheckFactories.registerCheck<readability::QualifiedAutoCheck>(
        "llvm-qualified-auto");
    CheckFactories.registerCheck<TwineLocalCheck>("llvm-twine-local");
  }

  ClangTidyOptions getModuleOptions() override {
    ClangTidyOptions Options;
    Options.CheckOptions["llvm-qualified-auto.AddConstToQualified"] = "false";
    Options.CheckOptions["llvm-else-after-return.WarnOnUnfixable"] = "false";
    Options.CheckOptions["llvm-else-after-return.WarnOnConditionVariables"] =
        "false";
    return Options;
  }
};

// Register the LLVMTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<LLVMModule> X("llvm-module",
                                                  "Adds LLVM lint checks.");

} // namespace llvm_check

// This anchor is used to force the linker to link in the generated object file
// and thus register the LLVMModule.
volatile int LLVMModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
