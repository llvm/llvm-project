//===--- MultipleNewInOneExpressionCheck.cpp - clang-tidy------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MultipleNewInOneExpressionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

// Determine if the result of an expression is "stored" in some way.
// It is true if the value is stored into a variable or used as initialization
// or passed to a function or constructor.
// For this use case compound assignments are not counted as a "store" (the 'E'
// expression should have pointer type).
static bool isExprValueStored(const Expr *E, ASTContext &C) {
  E = E->IgnoreParenCasts();
  // Get first non-paren, non-cast parent.
  ParentMapContext &PMap = C.getParentMapContext();
  DynTypedNodeList P = PMap.getParents(*E);
  if (P.size() != 1)
    return false;
  const Expr *ParentE = nullptr;
  while ((ParentE = P[0].get<Expr>()) && ParentE->IgnoreParenCasts() == E) {
    P = PMap.getParents(P[0]);
    if (P.size() != 1)
      return false;
  }

  if (const auto *ParentVarD = P[0].get<VarDecl>())
    return ParentVarD->getInit()->IgnoreParenCasts() == E;

  if (!ParentE)
    return false;

  if (const auto *BinOp = dyn_cast<BinaryOperator>(ParentE))
    return BinOp->getOpcode() == BO_Assign &&
           BinOp->getRHS()->IgnoreParenCasts() == E;

  return isa<CallExpr, CXXConstructExpr>(ParentE);
}

namespace {

AST_MATCHER_P(CXXTryStmt, hasHandlerFor,
              ast_matchers::internal::Matcher<QualType>, InnerMatcher) {
  for (unsigned NH = Node.getNumHandlers(), I = 0; I < NH; ++I) {
    const CXXCatchStmt *CatchS = Node.getHandler(I);
    // Check for generic catch handler (match anything).
    if (CatchS->getCaughtType().isNull())
      return true;
    ast_matchers::internal::BoundNodesTreeBuilder Result(*Builder);
    if (InnerMatcher.matches(CatchS->getCaughtType(), Finder, &Result)) {
      *Builder = std::move(Result);
      return true;
    }
  }
  return false;
}

AST_MATCHER(CXXNewExpr, mayThrow) {
  FunctionDecl *OperatorNew = Node.getOperatorNew();
  if (!OperatorNew)
    return false;
  return !OperatorNew->getType()->castAs<FunctionProtoType>()->isNothrow();
}

} // namespace

void MultipleNewInOneExpressionCheck::registerMatchers(MatchFinder *Finder) {
  auto BadAllocType =
      recordType(hasDeclaration(cxxRecordDecl(hasName("::std::bad_alloc"))));
  auto ExceptionType =
      recordType(hasDeclaration(cxxRecordDecl(hasName("::std::exception"))));
  auto BadAllocReferenceType = referenceType(pointee(BadAllocType));
  auto ExceptionReferenceType = referenceType(pointee(ExceptionType));

  auto CatchBadAllocType =
      qualType(hasCanonicalType(anyOf(BadAllocType, BadAllocReferenceType,
                                      ExceptionType, ExceptionReferenceType)));
  auto BadAllocCatchingTryBlock = cxxTryStmt(hasHandlerFor(CatchBadAllocType));

  auto NewExprMayThrow = cxxNewExpr(mayThrow());
  auto HasNewExpr1 = expr(anyOf(NewExprMayThrow.bind("new1"),
                                hasDescendant(NewExprMayThrow.bind("new1"))));
  auto HasNewExpr2 = expr(anyOf(NewExprMayThrow.bind("new2"),
                                hasDescendant(NewExprMayThrow.bind("new2"))));

  Finder->addMatcher(
      callExpr(
          hasAnyArgument(expr(HasNewExpr1).bind("arg1")),
          hasAnyArgument(
              expr(HasNewExpr2, unless(equalsBoundNode("arg1"))).bind("arg2")),
          hasAncestor(BadAllocCatchingTryBlock)),
      this);
  Finder->addMatcher(
      cxxConstructExpr(
          hasAnyArgument(expr(HasNewExpr1).bind("arg1")),
          hasAnyArgument(
              expr(HasNewExpr2, unless(equalsBoundNode("arg1"))).bind("arg2")),
          unless(isListInitialization()),
          hasAncestor(BadAllocCatchingTryBlock)),
      this);
  Finder->addMatcher(binaryOperator(hasLHS(HasNewExpr1), hasRHS(HasNewExpr2),
                                    unless(hasAnyOperatorName("&&", "||", ",")),
                                    hasAncestor(BadAllocCatchingTryBlock)),
                     this);
  Finder->addMatcher(
      cxxNewExpr(mayThrow(),
                 hasDescendant(NewExprMayThrow.bind("new2_in_new1")),
                 hasAncestor(BadAllocCatchingTryBlock))
          .bind("new1"),
      this);
}

void MultipleNewInOneExpressionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *NewExpr1 = Result.Nodes.getNodeAs<CXXNewExpr>("new1");
  const auto *NewExpr2 = Result.Nodes.getNodeAs<CXXNewExpr>("new2");
  const auto *NewExpr2InNewExpr1 =
      Result.Nodes.getNodeAs<CXXNewExpr>("new2_in_new1");
  if (!NewExpr2)
    NewExpr2 = NewExpr2InNewExpr1;
  assert(NewExpr1 && NewExpr2 && "Bound nodes not found.");

  // No warning if both allocations are not stored.
  // The value may be intentionally not stored (no deallocations needed or
  // self-destructing object).
  if (!isExprValueStored(NewExpr1, *Result.Context) &&
      !isExprValueStored(NewExpr2, *Result.Context))
    return;

  // In C++17 sequencing of a 'new' inside constructor arguments of another
  // 'new' is fixed. Still a leak can happen if the returned value from the
  // first 'new' is not saved (yet) and the second fails.
  if (getLangOpts().CPlusPlus17 && NewExpr2InNewExpr1)
    diag(NewExpr1->getBeginLoc(),
         "memory allocation may leak if an other allocation is sequenced after "
         "it and throws an exception")
        << NewExpr1->getSourceRange() << NewExpr2->getSourceRange();
  else
    diag(NewExpr1->getBeginLoc(),
         "memory allocation may leak if an other allocation is sequenced after "
         "it and throws an exception; order of these allocations is undefined")
        << NewExpr1->getSourceRange() << NewExpr2->getSourceRange();
}

} // namespace clang::tidy::bugprone
