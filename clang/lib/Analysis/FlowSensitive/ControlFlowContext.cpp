//===- ControlFlowContext.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a ControlFlowContext class that is used by dataflow
//  analyses that run over Control-Flow Graphs (CFGs).
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/ControlFlowContext.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"
#include <utility>

namespace clang {
namespace dataflow {

/// Returns a map from statements to basic blocks that contain them.
static llvm::DenseMap<const Stmt *, const CFGBlock *>
buildStmtToBasicBlockMap(const CFG &Cfg) {
  llvm::DenseMap<const Stmt *, const CFGBlock *> StmtToBlock;
  for (const CFGBlock *Block : Cfg) {
    if (Block == nullptr)
      continue;

    for (const CFGElement &Element : *Block) {
      auto Stmt = Element.getAs<CFGStmt>();
      if (!Stmt)
        continue;

      StmtToBlock[Stmt->getStmt()] = Block;
    }
    if (const Stmt *TerminatorStmt = Block->getTerminatorStmt())
      StmtToBlock[TerminatorStmt] = Block;
  }
  return StmtToBlock;
}

static llvm::BitVector findReachableBlocks(const CFG &Cfg) {
  llvm::BitVector BlockReachable(Cfg.getNumBlockIDs(), false);

  llvm::SmallVector<const CFGBlock *> BlocksToVisit;
  BlocksToVisit.push_back(&Cfg.getEntry());
  while (!BlocksToVisit.empty()) {
    const CFGBlock *Block = BlocksToVisit.back();
    BlocksToVisit.pop_back();

    if (BlockReachable[Block->getBlockID()])
      continue;

    BlockReachable[Block->getBlockID()] = true;

    for (const CFGBlock *Succ : Block->succs())
      if (Succ)
        BlocksToVisit.push_back(Succ);
  }

  return BlockReachable;
}

llvm::Expected<ControlFlowContext>
ControlFlowContext::build(const FunctionDecl &Func) {
  if (!Func.hasBody())
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Cannot analyze function without a body");

  return build(Func, *Func.getBody(), Func.getASTContext());
}

llvm::Expected<ControlFlowContext>
ControlFlowContext::build(const Decl &D, Stmt &S, ASTContext &C) {
  if (D.isTemplated())
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "Cannot analyze templated declarations");

  CFG::BuildOptions Options;
  Options.PruneTriviallyFalseEdges = true;
  Options.AddImplicitDtors = true;
  Options.AddTemporaryDtors = true;
  Options.AddInitializers = true;
  Options.AddCXXDefaultInitExprInCtors = true;

  // Ensure that all sub-expressions in basic blocks are evaluated.
  Options.setAllAlwaysAdd();

  auto Cfg = CFG::buildCFG(&D, &S, &C, Options);
  if (Cfg == nullptr)
    return llvm::createStringError(
        std::make_error_code(std::errc::invalid_argument),
        "CFG::buildCFG failed");

  llvm::DenseMap<const Stmt *, const CFGBlock *> StmtToBlock =
      buildStmtToBasicBlockMap(*Cfg);

  llvm::BitVector BlockReachable = findReachableBlocks(*Cfg);

  return ControlFlowContext(D, std::move(Cfg), std::move(StmtToBlock),
                            std::move(BlockReachable));
}

} // namespace dataflow
} // namespace clang
