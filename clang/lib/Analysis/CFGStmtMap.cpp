//===--- CFGStmtMap.h - Map from Stmt* to CFGBlock* -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the CFGStmtMap class, which defines a mapping from
//  Stmt* to CFGBlock*
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ParentMap.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CFGStmtMap.h"
#include <optional>

using namespace clang;

const CFGBlock *CFGStmtMap::getBlock(const Stmt *S) const {
  const Stmt *X = S;

  // If 'S' isn't in the map, walk the ParentMap to see if one of its ancestors
  // is in the map.
  while (X) {
    auto I = M.find(X);
    if (I != M.end())
      return I->second;
    X = PM->getParentIgnoreParens(X);
  }

  return nullptr;
}

CFGStmtMap::CFGStmtMap(const CFG &C, const ParentMap &PM) : PM(&PM) {
  // Walk all blocks, accumulating the block-level expressions, labels,
  // and terminators.
  for (const CFGBlock *B : C) {
    // First walk the block-level expressions.
    for (const CFGElement &CE : *B) {
      if (std::optional<CFGStmt> CS = CE.getAs<CFGStmt>())
        M.try_emplace(CS->getStmt(), B);
    }

    // Look at the label of the block.
    if (const Stmt *Label = B->getLabel())
      M[Label] = B;

    // Finally, look at the terminator.  If the terminator was already added
    // because it is a block-level expression in another block, overwrite
    // that mapping.
    if (const Stmt *Term = B->getTerminatorStmt())
      M[Term] = B;
  }
}
