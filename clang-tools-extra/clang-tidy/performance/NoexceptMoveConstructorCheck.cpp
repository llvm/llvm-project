//===--- NoexceptMoveConstructorCheck.cpp - clang-tidy---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NoexceptMoveConstructorCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace performance {

void NoexceptMoveConstructorCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matchers for C++11; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (!getLangOpts().CPlusPlus11)
    return;

  Finder->addMatcher(
      cxxMethodDecl(anyOf(cxxConstructorDecl(), hasOverloadedOperatorName("=")),
                    unless(isImplicit()), unless(isDeleted()))
          .bind("decl"),
      this);
}

void NoexceptMoveConstructorCheck::check(
    const MatchFinder::MatchResult &Result) {
  if (const auto *Decl = Result.Nodes.getNodeAs<CXXMethodDecl>("decl")) {
    StringRef MethodType = "assignment operator";
    if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(Decl)) {
      if (!Ctor->isMoveConstructor())
        return;
      MethodType = "constructor";
    } else if (!Decl->isMoveAssignmentOperator()) {
      return;
    }

    const auto *ProtoType = Decl->getType()->getAs<FunctionProtoType>();

    if (isUnresolvedExceptionSpec(ProtoType->getExceptionSpecType()))
      return;

    if (!isNoexceptExceptionSpec(ProtoType->getExceptionSpecType())) {
      diag(Decl->getLocation(), "move %0s should be marked noexcept")
          << MethodType;
      // FIXME: Add a fixit.
      return;
    }

    // Don't complain about nothrow(false), but complain on nothrow(expr)
    // where expr evaluates to false.
    if (ProtoType->canThrow() == CT_Can) {
      Expr *E = ProtoType->getNoexceptExpr();
      if (!isa<CXXBoolLiteralExpr>(ProtoType->getNoexceptExpr())) {
        diag(E->getExprLoc(),
             "noexcept specifier on the move %0 evaluates to 'false'")
            << MethodType;
      }
    }
  }
}

} // namespace performance
} // namespace tidy
} // namespace clang
