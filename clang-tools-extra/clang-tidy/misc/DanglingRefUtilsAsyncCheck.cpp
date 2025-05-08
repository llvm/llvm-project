//===--- DanglingRefUtilsAsyncCheck.cpp - clang-tidy ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DanglingRefUtilsAsyncCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

void DanglingRefUtilsAsyncCheck::registerMatchers(MatchFinder* Finder) {
  // if (0) {
  Finder->addMatcher(
      declRefExpr(
          hasParent(lambdaExpr(hasAncestor(
              expr(hasAncestor(cxxMemberCallExpr(has(memberExpr(
                       hasDeclaration(namedDecl(hasName("push_back"))),
                       has(declRefExpr(hasDeclaration(decl().bind("t1")))))))),
                   has(implicitCastExpr(has(declRefExpr(
                       hasDeclaration(namedDecl(hasName("Async"))))))))
                  .bind("x")))),
          hasDeclaration(decl().bind("decl")))
          .bind("root"),
      this);
  //}
  /*
  Finder->addMatcher(declRefExpr(hasParent(lambdaExpr(hasParent(
                                     expr(hasParent(expr().bind("call")))))))
                         .bind("root"),
                     this);
                     */
}

void DanglingRefUtilsAsyncCheck::check(const MatchFinder::MatchResult& Result) {
  const auto* Usage = Result.Nodes.getNodeAs<DeclRefExpr>("root");
  const auto* Tasks = Result.Nodes.getNodeAs<Decl>("t1");
  const auto* XDecl = Result.Nodes.getNodeAs<Decl>("decl");

  if (XDecl->getLocation() > Tasks->getLocation()) {
    // diag(Tasks->getLocation(), "Tasks allocation", DiagnosticIDs::Note);
    diag(Usage->getLocation(),
         "Might be a use-after-delete if exception is thrown inside lambda");
    // diag(XDecl->getLocation(), "Variable declaration");
  }
}

}  // namespace clang::tidy::misc
