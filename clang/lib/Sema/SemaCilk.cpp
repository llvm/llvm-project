//===--- SemaCilk.cpp - Semantic analysis for Cilk extensions -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for Cilk extensions.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExprCilk.h"
#include "clang/AST/StmtCilk.h"
#include "clang/Sema/SemaInternal.h"
using namespace clang;
using namespace sema;

static bool isValidCilkContext(Sema &S, SourceLocation Loc, StringRef Keyword) {
  // Cilk is not permitted in unevaluated contexts.
  if (S.isUnevaluatedContext()) {
    S.Diag(Loc, diag::err_cilk_unevaluated_context) << Keyword;
    return false;
  }

  // Any other usage must be within a function.
  FunctionDecl *FD = dyn_cast<FunctionDecl>(S.CurContext);
  if (!FD) {
    S.Diag(Loc, diag::err_cilk_outside_function) << Keyword;
    return false;
  }

  // A spawn cannot appear in a control scope.
  if (S.getCurScope()->getFlags() & Scope::ControlScope) {
    S.Diag(Loc, diag::err_spawn_invalid_scope) << Keyword;
    return false;
  }

  // TODO: Add more checks for the validity of the current context for Cilk.
  // (See isValidCoroutineContext for example code.)
  return true;
}

/// Check that this is a context in which a Cilk keywords can appear.
static FunctionScopeInfo *checkCilkContext(Sema &S, SourceLocation Loc,
                                           StringRef Keyword) {
  if (!isValidCilkContext(S, Loc, Keyword))
    return nullptr;

  assert(isa<FunctionDecl>(S.CurContext) && "not in a function scope");
  FunctionScopeInfo *ScopeInfo = S.getCurFunction();
  assert(ScopeInfo && "missing function scope for function");

  return ScopeInfo;
}

StmtResult
Sema::ActOnCilkSpawnStmt(SourceLocation SpawnLoc, Stmt *SubStmt) {
  if (!checkCilkContext(*this, SpawnLoc, "_Cilk_spawn"))
    return StmtError();

  DiagnoseUnusedExprResult(SubStmt);

  setFunctionHasBranchProtectedScope();

  PushFunctionScope();
  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);

  StmtResult Result = new (Context) CilkSpawnStmt(SpawnLoc, SubStmt);

  PopExpressionEvaluationContext();
  PopFunctionScopeInfo();

  return Result;
}

StmtResult
Sema::ActOnCilkSyncStmt(SourceLocation SyncLoc) {
  if (!checkCilkContext(*this, SyncLoc, "_Cilk_sync"))
    return StmtError();
  return new (Context) CilkSyncStmt(SyncLoc);
}

ExprResult Sema::ActOnCilkSpawnExpr(SourceLocation Loc, Expr *E) {
  FunctionScopeInfo *CilkCtx = checkCilkContext(*this, Loc, "_Cilk_spawn");
  if (!CilkCtx) {
    CorrectDelayedTyposInExpr(E);
    return ExprError();
  }

  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);

  if (E->getType()->isPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid()) return ExprError();
    E = R.get();
  }

  PopExpressionEvaluationContext();

  ExprResult Result =
    new (Context) CilkSpawnExpr(Loc, MaybeCreateExprWithCleanups(E));

  return Result;
}
