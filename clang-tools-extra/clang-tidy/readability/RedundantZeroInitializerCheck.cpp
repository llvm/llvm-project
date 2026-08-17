//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantZeroInitializerCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {
AST_MATCHER(InitListExpr, isSingleElementBracedList) {
  return Node.isExplicit() && Node.getNumInits() == 1;
}

AST_MATCHER(InitListExpr, isInMacro) {
  return Node.getBeginLoc().isMacroID() || Node.getEndLoc().isMacroID();
}

AST_MATCHER(InitListExpr, hasScalarArrayType) {
  const ConstantArrayType *CAT =
      Finder->getASTContext().getAsConstantArrayType(Node.getType());
  return CAT && CAT->getElementType()->isScalarType();
}

// Matches an array type location whose bound is deduced from the initializer
// (``T a[]``); replacing ``{0}`` with ``{}`` there would change the size.
AST_MATCHER(ArrayTypeLoc, hasDeducedArrayBound) {
  return Node.getSizeExpr() == nullptr;
}
} // namespace

void RedundantZeroInitializerCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      initListExpr(hasScalarArrayType(), isSingleElementBracedList(),
                   hasInit(0, ignoringParenImpCasts(integerLiteral(equals(0)))),
                   unless(isInMacro()),
                   unless(hasAncestor(cxxStdInitializerListExpr())),
                   unless(hasParent(varDecl(
                       hasTypeLoc(arrayTypeLoc(hasDeducedArrayBound()))))))
          .bind("init"),
      this);
}

void RedundantZeroInitializerCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *ILE = Result.Nodes.getNodeAs<InitListExpr>("init");
  const SourceRange Range = ILE->getSourceRange();
  diag(Range.getBegin(),
       "redundant zero initializer; replace with empty braces")
      << FixItHint::CreateReplacement(Range, "{}");
}

} // namespace clang::tidy::readability
