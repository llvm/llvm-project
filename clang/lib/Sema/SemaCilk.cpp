//===--- SemaCilk.cpp - Semantic analysis for Cilk extensions -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  // FunctionDecl *FD = cast<FunctionDecl>(S.CurContext);
  FunctionScopeInfo *ScopeInfo = S.getCurFunction();
  assert(ScopeInfo && "missing function scope for function");

  return ScopeInfo;
}

StmtResult
Sema::ActOnCilkSpawnStmt(SourceLocation SpawnLoc, Stmt *SubStmt) {
  if (!checkCilkContext(*this, SpawnLoc, "_Cilk_spawn"))
    return StmtError();

  DiagnoseUnusedExprResult(SubStmt);

  PushFunctionScope();
  // TODO: Figure out how to prevent jumps into and out of the spawned
  // substatement.
  setFunctionHasBranchProtectedScope();
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
  if (E->getType()->isPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid()) return ExprError();
    E = R.get();
  }

  return BuildCilkSpawnExpr(Loc, E);
}

ExprResult Sema::BuildCilkSpawnExpr(SourceLocation Loc, Expr *E) {
  FunctionScopeInfo *CilkCtx = checkCilkContext(*this, Loc, "_Cilk_spawn");
  if (!CilkCtx)
    return ExprError();

  if (E->getType()->isPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid()) return ExprError();
    E = R.get();
  }

  return new (Context) CilkSpawnExpr(Loc, E);
}
