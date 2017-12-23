//===--- OverloadedOperatorCheck.cpp - clang-tidy--------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OverloadedOperatorCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace fuchsia {

AST_MATCHER(FunctionDecl, isFuchsiaOverloadedOperator) {
  if (const auto *CXXMethodNode = dyn_cast<CXXMethodDecl>(&Node)) {
    if (CXXMethodNode->isCopyAssignmentOperator() ||
        CXXMethodNode->isMoveAssignmentOperator())
      return false;
  }
  return Node.isOverloadedOperator();
}

void OverloadedOperatorCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(functionDecl(isFuchsiaOverloadedOperator()).bind("decl"),
                     this);
}

void OverloadedOperatorCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *D = Result.Nodes.getNodeAs<FunctionDecl>("decl"))
    diag(D->getLocStart(), "cannot overload %0") << D;
}

} // namespace fuchsia
} // namespace tidy
} // namespace clang
