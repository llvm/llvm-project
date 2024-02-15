//===--- SemaOpenACC.cpp - Semantic Analysis for OpenACC constructs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for OpenACC constructs and
/// clauses.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/OpenACCKinds.h"
#include "clang/Sema/Sema.h"

using namespace clang;
bool Sema::ActOnOpenACCClause(OpenACCClauseKind ClauseKind,
                              SourceLocation StartLoc) {
  if (ClauseKind == OpenACCClauseKind::Invalid)
    return false;
  // For now just diagnose that it is unsupported and leave the parsing to do
  // whatever it can do. This function will eventually need to start returning
  // some sort of Clause AST type, but for now just return true/false based on
  // success.
  return Diag(StartLoc, diag::warn_acc_clause_unimplemented) << ClauseKind;
}
void Sema::ActOnOpenACCConstruct(OpenACCDirectiveKind K,
                                 SourceLocation StartLoc) {
  switch (K) {
  case OpenACCDirectiveKind::Invalid:
    // Nothing to do here, an invalid kind has nothing we can check here.  We
    // want to continue parsing clauses as far as we can, so we will just
    // ensure that we can still work and don't check any construct-specific
    // rules anywhere.
    break;
  default:
    Diag(StartLoc, diag::warn_acc_construct_unimplemented) << K;
    break;
  }
}

bool Sema::ActOnStartOpenACCStmtDirective(OpenACCDirectiveKind K,
                                          SourceLocation StartLoc) {
  return true;
}

StmtResult Sema::ActOnEndOpenACCStmtDirective(OpenACCDirectiveKind K,
                                              SourceLocation StartLoc,
                                              SourceLocation EndLoc,
                                              StmtResult AssocStmt) {
  return StmtEmpty();
}

StmtResult Sema::ActOnOpenACCAssociatedStmt(OpenACCDirectiveKind K,
                                            StmtResult AssocStmt) {
  return AssocStmt;
}

bool Sema::ActOnStartOpenACCDeclDirective(OpenACCDirectiveKind K,
                                          SourceLocation StartLoc) {
  return true;
}

DeclGroupRef Sema::ActOnEndOpenACCDeclDirective() { return DeclGroupRef{}; }
