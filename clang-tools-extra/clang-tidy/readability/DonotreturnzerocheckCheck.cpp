//===--- DonotreturnzerocheckCheck.cpp - clang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DonotreturnzerocheckCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {
bool isCPlusPlusOrC99(const LangOptions &LangOpts) {
  return LangOpts.CPlusPlus || LangOpts.C99;
}

void DonotreturnzerocheckCheck::registerMatchers(MatchFinder *Finder) {
  // FIXME: Add matchers.
  Finder->addMatcher(
      functionDecl(isMain(), returns(asString("int"))).bind("main"), this);
}

void DonotreturnzerocheckCheck::check(const MatchFinder::MatchResult &Result) {
  // FIXME: Add callback implementation.
  const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("main");
  if (isCPlusPlusOrC99(Result.Context->getLangOpts())) {
    SourceLocation ReturnLoc;
    if (MatchedDecl->hasBody()) {
      const CompoundStmt *Body = dyn_cast<CompoundStmt>(MatchedDecl->getBody());
      if (Body && !Body->body_empty()) {
        const Stmt *LastStmt = Body->body_back();
        if (const auto *Return = dyn_cast<ReturnStmt>(LastStmt)) {
          ReturnLoc = Return->getReturnLoc();
        }
      }
    }

    if (ReturnLoc.isValid()) {
      // Suggest removal of the redundant return statement.
      diag(ReturnLoc, "redundant 'return 0;' at the end of main")
          << FixItHint::CreateRemoval(
                 CharSourceRange::getTokenRange(ReturnLoc));
    }
  }
}

} // namespace clang::tidy::readability