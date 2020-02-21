//===--- SemaTapir.cpp - Semantic analysis for Tapir extensions -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for Tapir extensions.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtTapir.h"
#include "clang/Sema/SemaInternal.h"
using namespace clang;
using namespace sema;

StmtResult
Sema::ActOnSpawnStmt(SourceLocation SpawnLoc, StringRef sv, Stmt *SubStmt) {
  DiagnoseUnusedExprResult(SubStmt);

  PushFunctionScope();
  // TODO: Figure out how to prevent jumps into and out of the spawned
  // substatement.
  setFunctionHasBranchProtectedScope();
  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);

  StmtResult Result = new (Context) SpawnStmt(SpawnLoc, sv, SubStmt);

  PopExpressionEvaluationContext();
  PopFunctionScopeInfo();

  return Result;
}

StmtResult
Sema::ActOnSyncStmt(SourceLocation SyncLoc, StringRef sv) {
  return new (Context) SyncStmt(SyncLoc, sv);
}

