//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SwappedArgumentsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void SwappedArgumentsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(callExpr(unless(isInTemplateInstantiation())).bind("call"),
                     this);
}

/// Look through lvalue to rvalue and nop casts. This filters out
/// implicit conversions that have no effect on the input but block our view for
/// other implicit casts.
static const Expr *ignoreNoOpCasts(const Expr *E) {
  if (auto *Cast = dyn_cast<CastExpr>(E))
    if (Cast->getCastKind() == CK_LValueToRValue ||
        Cast->getCastKind() == CK_NoOp)
      return ignoreNoOpCasts(Cast->getSubExpr());
  return E;
}

/// Restrict the warning to implicit casts that are most likely
/// accidental. User defined or integral conversions fit in this category,
/// lvalue to rvalue or derived to base does not.
static bool isImplicitCastCandidate(const CastExpr *Cast) {
  return Cast->getCastKind() == CK_UserDefinedConversion ||
         Cast->getCastKind() == CK_FloatingToBoolean ||
         Cast->getCastKind() == CK_FloatingToIntegral ||
         Cast->getCastKind() == CK_IntegralToBoolean ||
         Cast->getCastKind() == CK_IntegralToFloating ||
         Cast->getCastKind() == CK_MemberPointerToBoolean ||
         Cast->getCastKind() == CK_PointerToBoolean ||
         (Cast->getCastKind() == CK_IntegralCast &&
          Cast->getSubExpr()->getType()->isBooleanType());
}

static bool areTypesSemiEqual(const QualType L, const QualType R) {
  if (L == R)
    return true;

  if (!L->isBuiltinType() || !R->isBuiltinType())
    return false;

  return (L->isFloatingType() && R->isFloatingType()) ||
         (L->isIntegerType() && R->isIntegerType()) ||
         (L->isBooleanType() && R->isBooleanType());
}

static bool areArgumentsPotentiallySwapped(const QualType LTo,
                                           const QualType RTo,
                                           const QualType LFrom,
                                           const QualType RFrom) {
  if (LTo == RTo || LFrom == RFrom)
    return false;

  const bool REq = areTypesSemiEqual(RTo, LFrom);
  if (LTo == RFrom && REq)
    return true;

  bool LEq = areTypesSemiEqual(LTo, RFrom);
  if (RTo == LFrom && LEq)
    return true;

  if (REq && LEq && !areTypesSemiEqual(RTo, LTo))
    return true;

  return false;
}

void SwappedArgumentsCheck::check(const MatchFinder::MatchResult &Result) {
  const ASTContext &Ctx = *Result.Context;
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");

  llvm::SmallPtrSet<const Expr *, 4> UsedArgs;
  for (unsigned I = 1, E = Call->getNumArgs(); I < E; ++I) {
    const Expr *LHS = Call->getArg(I - 1);
    const Expr *RHS = Call->getArg(I);

    // Only need to check RHS, as LHS has already been covered. We don't want to
    // emit two warnings for a single argument.
    if (UsedArgs.contains(RHS))
      continue;

    const auto *LHSCast = dyn_cast<ImplicitCastExpr>(ignoreNoOpCasts(LHS));
    const auto *RHSCast = dyn_cast<ImplicitCastExpr>(ignoreNoOpCasts(RHS));

    // Look if this is a potentially swapped argument pair. First look for
    // implicit casts.
    if (!LHSCast || !RHSCast || !isImplicitCastCandidate(LHSCast) ||
        !isImplicitCastCandidate(RHSCast))
      continue;

    // If the types that go into the implicit casts match the types of the other
    // argument in the declaration there is a high probability that the
    // arguments were swapped.
    // TODO: We could make use of the edit distance between the argument name
    // and the name of the passed variable in addition to this type based
    // heuristic.
    const Expr *LHSFrom = ignoreNoOpCasts(LHSCast->getSubExpr());
    const Expr *RHSFrom = ignoreNoOpCasts(RHSCast->getSubExpr());
    if (!areArgumentsPotentiallySwapped(LHS->getType(), RHS->getType(),
                                        LHSFrom->getType(), RHSFrom->getType()))
      continue;

    // Emit a warning and fix-its that swap the arguments.
    diag(Call->getBeginLoc(), "argument with implicit conversion from %0 "
                              "to %1 followed by argument converted from "
                              "%2 to %3, potentially swapped arguments.")
        << LHSFrom->getType() << LHS->getType() << RHSFrom->getType()
        << RHS->getType() << tooling::fixit::createReplacement(*LHS, *RHS, Ctx)
        << tooling::fixit::createReplacement(*RHS, *LHS, Ctx);

    // Remember that we emitted a warning for this argument.
    UsedArgs.insert(RHSCast);
  }
}

} // namespace clang::tidy::bugprone
