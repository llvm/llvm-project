//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StaticObjectExceptionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cert {

void StaticObjectExceptionCheck::registerMatchers(MatchFinder *Finder) {
  // Match any static or thread_local variable declaration that has an
  // initializer that can throw.
  Finder->addMatcher(
      traverse(
          TK_AsIs,
          varDecl(
              anyOf(hasThreadStorageDuration(), hasStaticStorageDuration()),
              unless(anyOf(isConstexpr(), hasType(cxxRecordDecl(isLambda())),
                           hasAncestor(functionDecl()))),
              anyOf(hasDescendant(cxxConstructExpr(hasDeclaration(
                        cxxConstructorDecl(unless(isNoThrow())).bind("func")))),
                    hasDescendant(cxxNewExpr(hasDeclaration(
                        functionDecl(unless(isNoThrow())).bind("func")))),
                    hasDescendant(callExpr(hasDeclaration(
                        functionDecl(unless(isNoThrow())).bind("func"))))))
              .bind("var")),
      this);
}

void StaticObjectExceptionCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *VD = Result.Nodes.getNodeAs<VarDecl>("var");
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");

  diag(VD->getLocation(),
       "initialization of %0 with %select{static|thread_local}1 storage "
       "duration may throw an exception that cannot be caught")
      << VD << (VD->getStorageDuration() == SD_Static ? 0 : 1);

  SourceLocation FuncLocation = Func->getLocation();
  if (FuncLocation.isValid()) {
    diag(FuncLocation,
         "possibly throwing %select{constructor|function}0 declared here",
         DiagnosticIDs::Note)
        << (isa<CXXConstructorDecl>(Func) ? 0 : 1);
  }
}

} // namespace clang::tidy::cert
