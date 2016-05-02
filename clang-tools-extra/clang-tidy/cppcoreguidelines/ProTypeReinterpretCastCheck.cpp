//===--- ProTypeReinterpretCastCheck.cpp - clang-tidy----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProTypeReinterpretCastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

void ProTypeReinterpretCastCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  Finder->addMatcher(cxxReinterpretCastExpr().bind("cast"), this);
}

void ProTypeReinterpretCastCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedCast =
      Result.Nodes.getNodeAs<CXXReinterpretCastExpr>("cast");
  diag(MatchedCast->getOperatorLoc(), "do not use reinterpret_cast");
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
