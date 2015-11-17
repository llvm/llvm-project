//===--- ThrownExceptionTypeCheck.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ThrownExceptionTypeCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace {
AST_MATCHER(CXXConstructorDecl, isNoThrowCopyConstructor) {
  if (!Node.isCopyConstructor())
    return false;

  if (const auto *FnTy = Node.getType()->getAs<FunctionProtoType>())
    return FnTy->isNothrow(Node.getASTContext());
  llvm_unreachable("Copy constructor with no prototype");
}
} // end namespace

namespace tidy {
void ThrownExceptionTypeCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  Finder->addMatcher(
      cxxThrowExpr(
          has(cxxConstructExpr(hasDeclaration(cxxConstructorDecl(
              isCopyConstructor(), unless(isNoThrowCopyConstructor()))))
          .bind("expr"))),
      this);

}

void ThrownExceptionTypeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *E = Result.Nodes.getNodeAs<Expr>("expr");
  diag(E->getExprLoc(),
       "thrown exception type is not nothrow copy constructible");
}

} // namespace tidy
} // namespace clang

