//===--- StmtOpenACC.cpp - Classes for OpenACC Constructs -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclasses of Stmt class declared in StmtOpenACC.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtOpenACC.h"
#include "clang/AST/ASTContext.h"
using namespace clang;

OpenACCComputeConstruct *
OpenACCComputeConstruct::CreateEmpty(const ASTContext &C, unsigned NumClauses) {
  void *Mem = C.Allocate(
      OpenACCComputeConstruct::totalSizeToAlloc<const OpenACCClause *>(
          NumClauses));
  auto *Inst = new (Mem) OpenACCComputeConstruct(NumClauses);
  return Inst;
}

OpenACCComputeConstruct *
OpenACCComputeConstruct::Create(const ASTContext &C, OpenACCDirectiveKind K,
                                SourceLocation BeginLoc, SourceLocation EndLoc,
                                ArrayRef<const OpenACCClause *> Clauses,
                                Stmt *StructuredBlock) {
  void *Mem = C.Allocate(
      OpenACCComputeConstruct::totalSizeToAlloc<const OpenACCClause *>(
          Clauses.size()));
  auto *Inst = new (Mem)
      OpenACCComputeConstruct(K, BeginLoc, EndLoc, Clauses, StructuredBlock);
  return Inst;
}
