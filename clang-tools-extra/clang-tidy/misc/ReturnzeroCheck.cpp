//===--- ReturnzeroCheck.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReturnzeroCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {
bool isCPlusPlusOrC99(const LangOptions &LangOpts) {
  return LangOpts.CPlusPlus || LangOpts.C99;
}

void ReturnzeroCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(isMain(), returns(asString("int"))).bind("main"), this);
}

void ReturnzeroCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("main");

  if (isCPlusPlusOrC99(Result.Context->getLangOpts())) {
    SourceLocation ReturnLoc;
    const Expr *RetValue;
    if (MatchedDecl->hasBody()) {
      const CompoundStmt *Body = dyn_cast<CompoundStmt>(MatchedDecl->getBody());
      if (Body && !Body->body_empty()) {
        const Stmt *LastStmt = Body->body_back();

        if (const auto *Return = dyn_cast<ReturnStmt>(LastStmt)) {
          ReturnLoc = Return->getReturnLoc();
          RetValue = Return->getRetValue();
        }
      }
    }

    if (ReturnLoc.isValid()) {
      if (RetValue->EvaluateKnownConstInt(*Result.Context).getSExtValue() ==
          0) {
        // Suggest removal of the redundant return statement.
        diag(ReturnLoc, "redundant 'return 0;' at the end of main")
            << FixItHint::CreateRemoval(
                   CharSourceRange::getTokenRange(ReturnLoc));
      }
    }
  }
}

} // namespace clang::tidy::misc
