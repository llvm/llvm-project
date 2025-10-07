//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnrollLoopsCheck.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <cmath>

using namespace clang::ast_matchers;

namespace clang::tidy::altera {

UnrollLoopsCheck::UnrollLoopsCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      MaxLoopIterations(Options.get("MaxLoopIterations", 100U)) {}

void UnrollLoopsCheck::registerMatchers(MatchFinder *Finder) {
  const auto HasLoopBound = hasDescendant(
      varDecl(matchesName("__end*"),
              hasDescendant(integerLiteral().bind("cxx_loop_bound"))));
  const auto CXXForRangeLoop =
      cxxForRangeStmt(anyOf(HasLoopBound, unless(HasLoopBound)));
  const auto AnyLoop = anyOf(forStmt(), whileStmt(), doStmt(), CXXForRangeLoop);
  Finder->addMatcher(
      stmt(AnyLoop, unless(hasDescendant(stmt(AnyLoop)))).bind("loop"), this);
}

void UnrollLoopsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Loop = Result.Nodes.getNodeAs<Stmt>("loop");
  const auto *CXXLoopBound =
      Result.Nodes.getNodeAs<IntegerLiteral>("cxx_loop_bound");
  const ASTContext *Context = Result.Context;
  switch (unrollType(Loop, Result.Context)) {
  case NotUnrolled:
    diag(Loop->getBeginLoc(),
         "kernel performance could be improved by unrolling this loop with a "
         "'#pragma unroll' directive");
    break;
  case PartiallyUnrolled:
    // Loop already partially unrolled, do nothing.
    break;
  case FullyUnrolled:
    if (hasKnownBounds(Loop, CXXLoopBound, Context)) {
      if (hasLargeNumIterations(Loop, CXXLoopBound, Context)) {
        diag(Loop->getBeginLoc(),
             "loop likely has a large number of iterations and thus "
             "cannot be fully unrolled; to partially unroll this loop, use "
             "the '#pragma unroll <num>' directive");
        return;
      }
      return;
    }
    if (isa<WhileStmt, DoStmt>(Loop)) {
      diag(Loop->getBeginLoc(),
           "full unrolling requested, but loop bounds may not be known; to "
           "partially unroll this loop, use the '#pragma unroll <num>' "
           "directive",
           DiagnosticIDs::Note);
      break;
    }
    diag(Loop->getBeginLoc(),
         "full unrolling requested, but loop bounds are not known; to "
         "partially unroll this loop, use the '#pragma unroll <num>' "
         "directive");
    break;
  }
}

enum UnrollLoopsCheck::UnrollType
UnrollLoopsCheck::unrollType(const Stmt *Statement, ASTContext *Context) {
  const DynTypedNodeList Parents = Context->getParents<Stmt>(*Statement);
  for (const DynTypedNode &Parent : Parents) {
    const auto *ParentStmt = Parent.get<AttributedStmt>();
    if (!ParentStmt)
      continue;
    for (const Attr *Attribute : ParentStmt->getAttrs()) {
      const auto *LoopHint = dyn_cast<LoopHintAttr>(Attribute);
      if (!LoopHint)
        continue;
      switch (LoopHint->getState()) {
      case LoopHintAttr::Numeric:
        return PartiallyUnrolled;
      case LoopHintAttr::Disable:
        return NotUnrolled;
      case LoopHintAttr::Full:
        return FullyUnrolled;
      case LoopHintAttr::Enable:
        return FullyUnrolled;
      case LoopHintAttr::AssumeSafety:
        return NotUnrolled;
      case LoopHintAttr::FixedWidth:
        return NotUnrolled;
      case LoopHintAttr::ScalableWidth:
        return NotUnrolled;
      }
    }
  }
  return NotUnrolled;
}

bool UnrollLoopsCheck::hasKnownBounds(const Stmt *Statement,
                                      const IntegerLiteral *CXXLoopBound,
                                      const ASTContext *Context) {
  if (isa<CXXForRangeStmt>(Statement))
    return CXXLoopBound != nullptr;
  // Too many possibilities in a while statement, so always recommend partial
  // unrolling for these.
  if (isa<WhileStmt, DoStmt>(Statement))
    return false;
  // The last loop type is a for loop.
  const auto *ForLoop = cast<ForStmt>(Statement);
  const Stmt *Initializer = ForLoop->getInit();
  const Expr *Conditional = ForLoop->getCond();
  const Expr *Increment = ForLoop->getInc();
  if (!Initializer || !Conditional || !Increment)
    return false;
  // If the loop variable value isn't known, loop bounds are unknown.
  if (const auto *InitDeclStatement = dyn_cast<DeclStmt>(Initializer)) {
    if (const auto *VariableDecl =
            dyn_cast<VarDecl>(InitDeclStatement->getSingleDecl())) {
      APValue *Evaluation = VariableDecl->evaluateValue();
      if (!Evaluation || !Evaluation->hasValue())
        return false;
    }
  }
  // If increment is unary and not one of ++ and --, loop bounds are unknown.
  if (const auto *Op = dyn_cast<UnaryOperator>(Increment))
    if (!Op->isIncrementDecrementOp())
      return false;

  if (const auto *BinaryOp = dyn_cast<BinaryOperator>(Conditional)) {
    const Expr *LHS = BinaryOp->getLHS();
    const Expr *RHS = BinaryOp->getRHS();
    // If both sides are value dependent or constant, loop bounds are unknown.
    return LHS->isEvaluatable(*Context) != RHS->isEvaluatable(*Context);
  }
  return false; // If it's not a binary operator, loop bounds are unknown.
}

const Expr *UnrollLoopsCheck::getCondExpr(const Stmt *Statement) {
  if (const auto *ForLoop = dyn_cast<ForStmt>(Statement))
    return ForLoop->getCond();
  if (const auto *WhileLoop = dyn_cast<WhileStmt>(Statement))
    return WhileLoop->getCond();
  if (const auto *DoWhileLoop = dyn_cast<DoStmt>(Statement))
    return DoWhileLoop->getCond();
  if (const auto *CXXRangeLoop = dyn_cast<CXXForRangeStmt>(Statement))
    return CXXRangeLoop->getCond();
  llvm_unreachable("Unknown loop");
}

bool UnrollLoopsCheck::hasLargeNumIterations(const Stmt *Statement,
                                             const IntegerLiteral *CXXLoopBound,
                                             const ASTContext *Context) {
  // Because hasKnownBounds is called before this, if this is true, then
  // CXXLoopBound is also matched.
  if (isa<CXXForRangeStmt>(Statement)) {
    assert(CXXLoopBound && "CXX ranged for loop has no loop bound");
    return exprHasLargeNumIterations(CXXLoopBound, Context);
  }
  const auto *ForLoop = cast<ForStmt>(Statement);
  const Stmt *Initializer = ForLoop->getInit();
  const Expr *Conditional = ForLoop->getCond();
  const Expr *Increment = ForLoop->getInc();
  int InitValue = 0;
  // If the loop variable value isn't known, we can't know the loop bounds.
  if (const auto *InitDeclStatement = dyn_cast<DeclStmt>(Initializer)) {
    if (const auto *VariableDecl =
            dyn_cast<VarDecl>(InitDeclStatement->getSingleDecl())) {
      APValue *Evaluation = VariableDecl->evaluateValue();
      if (!Evaluation || !Evaluation->isInt())
        return true;
      InitValue = Evaluation->getInt().getExtValue();
    }
  }

  int EndValue = 0;
  const auto *BinaryOp = cast<BinaryOperator>(Conditional);
  if (!extractValue(EndValue, BinaryOp, Context))
    return true;

  double Iterations = 0.0;

  // If increment is unary and not one of ++, --, we can't know the loop bounds.
  if (const auto *Op = dyn_cast<UnaryOperator>(Increment)) {
    if (Op->isIncrementOp())
      Iterations = EndValue - InitValue;
    else if (Op->isDecrementOp())
      Iterations = InitValue - EndValue;
    else
      llvm_unreachable("Unary operator neither increment nor decrement");
  }

  // If increment is binary and not one of +, -, *, /, we can't know the loop
  // bounds.
  if (const auto *Op = dyn_cast<BinaryOperator>(Increment)) {
    int ConstantValue = 0;
    if (!extractValue(ConstantValue, Op, Context))
      return true;
    switch (Op->getOpcode()) {
    case (BO_AddAssign):
      Iterations = std::ceil(float(EndValue - InitValue) / ConstantValue);
      break;
    case (BO_SubAssign):
      Iterations = std::ceil(float(InitValue - EndValue) / ConstantValue);
      break;
    case (BO_MulAssign):
      Iterations =
          1 + (std::log((double)EndValue) - std::log((double)InitValue)) /
                  std::log((double)ConstantValue);
      break;
    case (BO_DivAssign):
      Iterations =
          1 + (std::log((double)InitValue) - std::log((double)EndValue)) /
                  std::log((double)ConstantValue);
      break;
    default:
      // All other operators are not handled; assume large bounds.
      return true;
    }
  }
  return Iterations > MaxLoopIterations;
}

bool UnrollLoopsCheck::extractValue(int &Value, const BinaryOperator *Op,
                                    const ASTContext *Context) {
  const Expr *LHS = Op->getLHS();
  const Expr *RHS = Op->getRHS();
  Expr::EvalResult Result;
  if (LHS->isEvaluatable(*Context))
    LHS->EvaluateAsRValue(Result, *Context);
  else if (RHS->isEvaluatable(*Context))
    RHS->EvaluateAsRValue(Result, *Context);
  else
    return false; // Cannot evaluate either side.
  if (!Result.Val.isInt())
    return false; // Cannot check number of iterations, return false to be
                  // safe.
  Value = Result.Val.getInt().getExtValue();
  return true;
}

bool UnrollLoopsCheck::exprHasLargeNumIterations(
    const Expr *Expression, const ASTContext *Context) const {
  Expr::EvalResult Result;
  if (Expression->EvaluateAsRValue(Result, *Context)) {
    if (!Result.Val.isInt())
      return false; // Cannot check number of iterations, return false to be
                    // safe.
    // The following assumes values go from 0 to Val in increments of 1.
    return Result.Val.getInt() > MaxLoopIterations;
  }
  // Cannot evaluate Expression as an r-value, so cannot check number of
  // iterations.
  return false;
}

void UnrollLoopsCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "MaxLoopIterations", MaxLoopIterations);
}

} // namespace clang::tidy::altera
