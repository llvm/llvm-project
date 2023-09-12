//===--- MoveSmartPointerContentsCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "MoveSmartPointerContentsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

bool MoveSmartPointerContentsCheck::isLanguageVersionSupported(
    const LangOptions &LangOptions) const {
  return LangOptions.CPlusPlus11;
}

MoveSmartPointerContentsCheck::MoveSmartPointerContentsCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      UniquePointerClasses(utils::options::parseStringList(
          Options.get("UniquePointerClasses", "std::unique_ptr"))) {}

void MoveSmartPointerContentsCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "UniquePtrClasses",
                utils::options::serializeStringList(UniquePointerClasses));
}

void MoveSmartPointerContentsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(
          callee(functionDecl(hasName("std::move"))),
          hasArgument(0, cxxOperatorCallExpr(hasOverloadedOperatorName("*"),
                                             callee(cxxMethodDecl(ofClass(
                                                 matchers::matchesAnyListedName(
                                                     UniquePointerClasses)))))))
          .bind("call"),
      this);
}

void MoveSmartPointerContentsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");

  if (Call) {
    diag(Call->getBeginLoc(),
         "prefer to move the smart pointer rather than its contents");
  }
}

} // namespace clang::tidy::performance
