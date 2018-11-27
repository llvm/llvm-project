//===--- StaticallyConstructedObjectsCheck.cpp - clang-tidy----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

  // Constexpr requires C++11 or later.
  if (!getLangOpts().CPlusPlus11)
    return;

  Finder->addMatcher(varDecl(
                         // Match global, statically stored objects...
                         isGlobalStatic(),
                         // ... that have C++ constructors...
                         hasDescendant(cxxConstructExpr(unless(allOf(
                             // ... unless it is constexpr ...
                             hasDeclaration(cxxConstructorDecl(isConstexpr())),
                             // ... and is statically initialized.
                             isConstantInitializer())))))
                         .bind("decl"),
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
