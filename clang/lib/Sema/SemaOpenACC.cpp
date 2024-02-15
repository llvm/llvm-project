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

#include "clang/Basic/OpenACCKinds.h"
#include "clang/Sema/Sema.h"

using namespace clang;
bool Sema::ActOnOpenACCClause(OpenACCClauseKind ClauseKind,
                              SourceLocation StartLoc) {
  return true;
}
void Sema::ActOnOpenACCConstruct(OpenACCDirectiveKind K,
                                 SourceLocation StartLoc) {}

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
