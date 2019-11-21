//===--- ImplicitConversionInLoopCheck.cpp - clang-tidy--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImplicitConversionInLoopCheck.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace performance {

// Checks if the stmt is a ImplicitCastExpr with a CastKind that is not a NoOp.
// The subtelty is that in some cases (user defined conversions), we can
// get to ImplicitCastExpr inside each other, with the outer one a NoOp. In this
// case we skip the first cast expr.
static bool IsNonTrivialImplicitCast(const Stmt *ST) {
  if (const auto *ICE = dyn_cast<ImplicitCastExpr>(ST)) {
    return (ICE->getCastKind() != CK_NoOp) ||
            IsNonTrivialImplicitCast(ICE->getSubExpr());
  }
  return false;
}

void ImplicitConversionInLoopCheck::registerMatchers(MatchFinder *Finder) {
  // We look for const ref loop variables that (optionally inside an
  // ExprWithCleanup) materialize a temporary, and contain a implicit
  // conversion. The check on the implicit conversion is done in check() because
  // we can't access implicit conversion subnode via matchers: has() skips casts
  // and materialize! We also bind on the call to operator* to get the proper
  // type in the diagnostic message. We use both cxxOperatorCallExpr for user
  // defined operator and unaryOperator when the iterator is a pointer, like
  // for arrays or std::array.
  //
  // Note that when the implicit conversion is done through a user defined
  // conversion operator, the node is a CXXMemberCallExpr, not a
  // CXXOperatorCallExpr, so it should not get caught by the
  // cxxOperatorCallExpr() matcher.
  Finder->addMatcher(
      cxxForRangeStmt(hasLoopVariable(
          varDecl(
              hasType(qualType(references(qualType(isConstQualified())))),
              hasInitializer(
                  expr(anyOf(hasDescendant(
                                 cxxOperatorCallExpr().bind("operator-call")),
                             hasDescendant(unaryOperator(hasOperatorName("*"))
                                               .bind("operator-call"))))
                      .bind("init")))
              .bind("faulty-var"))),
      this);
}

void ImplicitConversionInLoopCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *VD = Result.Nodes.getNodeAs<VarDecl>("faulty-var");
  const auto *Init = Result.Nodes.getNodeAs<Expr>("init");
  const auto *OperatorCall =
      Result.Nodes.getNodeAs<Expr>("operator-call");

  if (const auto *Cleanup = dyn_cast<ExprWithCleanups>(Init))
    Init = Cleanup->getSubExpr();

  const auto *Materialized = dyn_cast<MaterializeTemporaryExpr>(Init);
  if (!Materialized)
    return;

  // We ignore NoOp casts. Those are generated if the * operator on the
  // iterator returns a value instead of a reference, and the loop variable
  // is a reference. This situation is fine (it probably produces the same
  // code at the end).
  if (IsNonTrivialImplicitCast(Materialized->getSubExpr()))
    ReportAndFix(Result.Context, VD, OperatorCall);
}

void ImplicitConversionInLoopCheck::ReportAndFix(
    const ASTContext *Context, const VarDecl *VD,
    const Expr *OperatorCall) {
  // We only match on const ref, so we should print a const ref version of the
  // type.
  QualType ConstType = OperatorCall->getType().withConst();
  QualType ConstRefType = Context->getLValueReferenceType(ConstType);
  const char Message[] =
      "the type of the loop variable %0 is different from the one returned "
      "by the iterator and generates an implicit conversion; you can either "
      "change the type to the matching one (%1 but 'const auto&' is always a "
      "valid option) or remove the reference to make it explicit that you are "
      "creating a new value";
  diag(VD->getBeginLoc(), Message) << VD << ConstRefType;
}

} // namespace performance
} // namespace tidy
} // namespace clang
