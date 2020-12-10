//===--- StaticallyConstructedObjectsCheck.cpp - clang-tidy----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StaticallyConstructedObjectsCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace fuchsia {

namespace {
AST_MATCHER(Expr, isConstantInitializer) {
  return Node.isConstantInitializer(Finder->getASTContext(), false);
}

AST_MATCHER(VarDecl, isGlobalStatic) {
  return Node.getStorageDuration() == SD_Static && !Node.isLocalVarDecl();
}
} // namespace

void StaticallyConstructedObjectsCheck::registerMatchers(MatchFinder *Finder) {
  // Constructing global, non-trivial objects with static storage is
  // disallowed, unless the object is statically initialized with a constexpr
  // constructor or has no explicit constructor.
  Finder->addMatcher(
      traverse(TK_AsIs,
               varDecl(
                   // Match global, statically stored objects...
                   isGlobalStatic(),
                   // ... that have C++ constructors...
                   hasDescendant(cxxConstructExpr(unless(allOf(
                       // ... unless it is constexpr ...
                       hasDeclaration(cxxConstructorDecl(isConstexpr())),
                       // ... and is statically initialized.
                       isConstantInitializer())))))
                   .bind("decl")),
      this);
}

void StaticallyConstructedObjectsCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *D = Result.Nodes.getNodeAs<VarDecl>("decl"))
    diag(D->getBeginLoc(), "static objects are disallowed; if possible, use a "
                           "constexpr constructor instead");
}

} // namespace fuchsia
} // namespace tidy
} // namespace clang
