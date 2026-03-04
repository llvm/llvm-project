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
    Stmt *AssociatedStatement, bool NoRemainder, bool MaskPostlude,
    ThreadScheduleKind ThreadKind, ValueDecl *ThreadChunk,
    std::optional<uint64_t> ThreadChunkVal) {
  void *Mem = C.Allocate(
      RippleComputeConstruct::totalSizeToAlloc<uint64_t>(Dims.size()));
  auto *Inst = new (Mem) RippleComputeConstruct(
      PragmaLoc, BlockShapeLoc, DimsLoc, BlockShape, Dims, AssociatedStatement,
      NoRemainder, MaskPostlude, ThreadKind, ThreadChunk, ThreadChunkVal);
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

std::pair<ForStmt *, ForStmt *>
RippleComputeConstruct::getInnerThreadLoops() const {
  if (!threadCodegen() || !getRippleLoopStmt())
    return {nullptr, nullptr};
  auto OuterLoopBody = cast<CompoundStmt>(getRippleLoopStmt()->getBody());
  // Look for: if (...) for { full_iterations } else for { masked_remainder }
  auto Res = llvm::find_if(OuterLoopBody->body(), [](auto *S) {
    if (auto *If = dyn_cast<IfStmt>(S))
      return isa_and_present<ForStmt>(If->getThen()) &&
             isa_and_present<ForStmt>(If->getElse());
    else
      return false;
  });
  if (Res != OuterLoopBody->body_end()) {
    auto *If = cast<IfStmt>(*Res);
    return {cast<ForStmt>(If->getThen()), cast<ForStmt>(If->getElse())};
  }
  return {nullptr, nullptr};
}

void RippleComputeConstruct::setInnerThreadLoops(
    std::pair<Stmt *, Stmt *> NewInnerStmts) {
  if (!threadCodegen() || !getRippleLoopStmt())
    return;
  auto OuterLoopBody = cast<CompoundStmt>(getRippleLoopStmt()->getBody());
  // Look for: if (...) for { full_iterations } else for { masked_remainder }
  auto Res = llvm::find_if(OuterLoopBody->body(), [](auto *S) {
    if (auto *If = dyn_cast<IfStmt>(S))
      return isa_and_present<ForStmt>(If->getThen()) &&
             isa_and_present<ForStmt>(If->getElse());
    else
      return false;
  });
  if (Res == OuterLoopBody->body_end())
    llvm_unreachable("ripple thread loop expects a dispatch IfStmt");
  auto *If = cast<IfStmt>(*Res); // we searched for an IfStmt
  If->setThen(NewInnerStmts.first);
  If->setElse(NewInnerStmts.second);
}
