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

#ifndef LLVM_CLANG_ANALYSIS_CFGSTMTMAP_H
#define LLVM_CLANG_ANALYSIS_CFGSTMTMAP_H

#include "clang/Analysis/CFG.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {

class ParentMap;
class Stmt;

class CFGStmtMap {
  const ParentMap *PM;
  llvm::DenseMap<const Stmt *, const CFGBlock *> M;

public:
  CFGStmtMap(const CFG &C, const ParentMap &PM);

  /// Returns the CFGBlock the specified Stmt* appears in.  For Stmt* that
  /// are terminators, the CFGBlock is the block they appear as a terminator,
  /// and not the block they appear as a block-level expression (e.g, '&&').
  /// CaseStmts and LabelStmts map to the CFGBlock they label.
  const CFGBlock *getBlock(const Stmt *S) const;
};

} // end clang namespace
#endif
