//===--- FloatLoopCounter.cpp - clang-tidy---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FloatLoopCounter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cert {

void FloatLoopCounter::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      forStmt(hasIncrement(expr(hasType(realFloatingPointType())))).bind("for"),
      this);
}

void FloatLoopCounter::check(const MatchFinder::MatchResult &Result) {
  const auto *FS = Result.Nodes.getNodeAs<ForStmt>("for");

  diag(FS->getInc()->getExprLoc(), "loop induction expression should not have "
                                   "floating-point type");
}

} // namespace cert
} // namespace tidy
} // namespace clang
