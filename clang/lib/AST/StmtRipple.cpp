//===--- StmtRipple.cpp - Classes for Ripple Constructs -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the subclasses of Stmt class declared in StmtRipple.h
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtRipple.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <iterator>

using namespace clang;

RippleComputeConstruct *RippleComputeConstruct::Create(
    const ASTContext &C, SourceRange PragmaLoc, SourceRange BlockShapeLoc,
    SourceRange DimsLoc, ValueDecl *BlockShape, ArrayRef<uint64_t> Dims,
    ForStmt *AssociatedLoop, bool NoRemainder, bool MaskPostlude, bool IsThread,
    ValueDecl *ThreadChunk, std::optional<uint64_t> ThreadChunkVal) {
  void *Mem = C.Allocate(
      RippleComputeConstruct::totalSizeToAlloc<uint64_t>(Dims.size()));
  auto *Inst = new (Mem) RippleComputeConstruct(
      PragmaLoc, BlockShapeLoc, DimsLoc, BlockShape, Dims, AssociatedLoop,
      NoRemainder, MaskPostlude, IsThread, ThreadChunk, ThreadChunkVal);
  return Inst;
}

RippleComputeConstruct *RippleComputeConstruct::CreateEmpty(const ASTContext &C,
                                                            uint64_t NumDims) {
  void *Mem =
      C.Allocate(RippleComputeConstruct::totalSizeToAlloc<uint64_t>(NumDims));
  auto *Inst = new (Mem) RippleComputeConstruct(NumDims);
  return Inst;
}

SmallVector<const VarDecl *, RippleComputeConstruct::NumVarDecls>
RippleComputeConstruct::getRippleVarDecls() const {
  SmallVector<const VarDecl *, NumVarDecls> Decls;
  // Either they are all present or none are present!
  if (!getAssociatedLoopIters())
    return Decls;

  for (int Decl = FirstVarDecl; Decl <= LastVarDecl; Decl++)
    if (auto *DRE = cast_if_present<DeclRefExpr>(SubStmts[Decl]))
      if (VarDecl *VD = cast<VarDecl>(DRE->getDecl()))
        Decls.push_back(VD);

  if (threadCodegen())
    // LOOP_IV_ORIGIN may be nullptr and STEP_RIPPLE, LOOP_INIT_RIPPLE aren't
    // used in thread mode
    assert(Decls.size() == NumVarDecls - 2 || Decls.size() == NumVarDecls - 3);
  else
    // LOOP_IV_ORIGIN may be nullptr and THREAD_CHUNK_SIZE isn't used in vector
    // mode
    assert(Decls.size() == NumVarDecls - 1 || Decls.size() == NumVarDecls - 2);

  return Decls;
}

void RippleComputeConstruct::printPragma(raw_ostream &OS) const {
  auto DimIds = getDimensionIds();
  OS << "#pragma ripple parallel Block(" << getBlockShape()->getName()
     << ") Dims(" << DimIds[0];

  for (auto DimId : llvm::make_range(std::next(DimIds.begin()), DimIds.end())) {
    OS << ", " << DimId;
  }
  OS << ")";
  if (!generateRemainder()) {
    OS << " NoRemainder";
  }
}

void RippleComputeConstruct::print(raw_ostream &OS) const {
  printPragma(OS);
  auto *For = getAssociatedForStmt();
  OS << " -> for (" << For->getInit() << "; " << For->getCond() << "; "
     << For->getInc() << ")";
}
