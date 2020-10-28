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
#include "../readability/BracesAroundStatementsCheck.h"
#include "../readability/FunctionSizeCheck.h"
#include "../readability/NamespaceCommentCheck.h"
#include "AvoidCStyleCastsCheck.h"
#include "AvoidNSObjectNewCheck.h"
#include "AvoidThrowingObjCExceptionCheck.h"
#include "AvoidUnderscoreInGoogletestNameCheck.h"
#include "DefaultArgumentsCheck.h"
#include "ExplicitConstructorCheck.h"
#include "ExplicitMakePairCheck.h"
#include "FunctionNamingCheck.h"
#include "GlobalNamesInHeadersCheck.h"
#include "GlobalVariableDeclarationCheck.h"
#include "IntegerTypesCheck.h"
#include "OverloadedUnaryAndCheck.h"
#include "TodoCommentCheck.h"
#include "UnnamedNamespaceInHeaderCheck.h"
#include "UpgradeGoogletestCaseCheck.h"
#include "UsingNamespaceDirectiveCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace google {

class GoogleModule : public ClangTidyModule {
 public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<build::ExplicitMakePairCheck>(
        "google-build-explicit-make-pair");
    CheckFactories.registerCheck<build::UnnamedNamespaceInHeaderCheck>(
        "google-build-namespaces");
    CheckFactories.registerCheck<build::UsingNamespaceDirectiveCheck>(
        "google-build-using-namespace");
    CheckFactories.registerCheck<DefaultArgumentsCheck>(
        "google-default-arguments");
    CheckFactories.registerCheck<ExplicitConstructorCheck>(
        "google-explicit-constructor");
    CheckFactories.registerCheck<readability::GlobalNamesInHeadersCheck>(
        "google-global-names-in-headers");
    CheckFactories.registerCheck<objc::AvoidNSObjectNewCheck>(
        "google-objc-avoid-nsobject-new");
    CheckFactories.registerCheck<objc::AvoidThrowingObjCExceptionCheck>(
        "google-objc-avoid-throwing-exception");
    CheckFactories.registerCheck<objc::FunctionNamingCheck>(
        "google-objc-function-naming");
    CheckFactories.registerCheck<objc::GlobalVariableDeclarationCheck>(
        "google-objc-global-variable-declaration");
    CheckFactories.registerCheck<runtime::IntegerTypesCheck>(
        "google-runtime-int");
    CheckFactories.registerCheck<runtime::OverloadedUnaryAndCheck>(
        "google-runtime-operator");
    CheckFactories
        .registerCheck<readability::AvoidUnderscoreInGoogletestNameCheck>(
            "google-readability-avoid-underscore-in-googletest-name");
    CheckFactories.registerCheck<readability::AvoidCStyleCastsCheck>(
        "google-readability-casting");
    CheckFactories.registerCheck<readability::TodoCommentCheck>(
        "google-readability-todo");
    CheckFactories
        .registerCheck<clang::tidy::readability::BracesAroundStatementsCheck>(
            "google-readability-braces-around-statements");
    CheckFactories.registerCheck<clang::tidy::readability::FunctionSizeCheck>(
        "google-readability-function-size");
    CheckFactories
        .registerCheck<clang::tidy::readability::NamespaceCommentCheck>(
            "google-readability-namespace-comments");
    CheckFactories.registerCheck<UpgradeGoogletestCaseCheck>(
        "google-upgrade-googletest-case");
  }

  ClangTidyOptions getModuleOptions() override {
    ClangTidyOptions Options;
    auto &Opts = Options.CheckOptions;
    Opts["google-readability-braces-around-statements.ShortStatementLines"] =
        "1";
    Opts["google-readability-function-size.StatementThreshold"] = "800";
    Opts["google-readability-namespace-comments.ShortNamespaceLines"] = "10";
    Opts["google-readability-namespace-comments.SpacesBeforeComments"] = "2";
    return Options;
  }
};

// Register the GoogleTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<GoogleModule> X("google-module",
                                                    "Adds Google lint checks.");

}  // namespace google

// This anchor is used to force the linker to link in the generated object file
// and thus register the GoogleModule.
volatile int GoogleModuleAnchorSource = 0;

}  // namespace tidy
}  // namespace clang
