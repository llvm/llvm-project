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
OpenACCComputeConstruct::CreateEmpty(const ASTContext &C, EmptyShell) {
  void *Mem = C.Allocate(sizeof(OpenACCComputeConstruct),
                         alignof(OpenACCComputeConstruct));
  auto *Inst = new (Mem) OpenACCComputeConstruct;
  return Inst;
}

OpenACCComputeConstruct *
OpenACCComputeConstruct::Create(const ASTContext &C, OpenACCDirectiveKind K,
                                SourceLocation BeginLoc, SourceLocation EndLoc,
                                Stmt *StructuredBlock) {
  void *Mem = C.Allocate(sizeof(OpenACCComputeConstruct),
                         alignof(OpenACCComputeConstruct));
  auto *Inst =
      new (Mem) OpenACCComputeConstruct(K, BeginLoc, EndLoc, StructuredBlock);
  return Inst;
}
