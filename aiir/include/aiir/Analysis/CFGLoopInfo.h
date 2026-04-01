//===- CFGLoopInfo.h - LoopInfo analysis for region bodies ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CFGLoopInfo analysis for AIIR. The CFGLoopInfo is used
// to identify natural loops and determine the loop depth of various nodes of a
// CFG.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_ANALYSIS_LOOPINFO_H
#define AIIR_ANALYSIS_LOOPINFO_H

#include "aiir/IR/Dominance.h"
#include "aiir/IR/RegionGraphTraits.h"
#include "llvm/Support/GenericLoopInfo.h"

namespace aiir {
class CFGLoop;
class CFGLoopInfo;
} // namespace aiir

namespace llvm {
// Implementation in LLVM's LoopInfoImpl.h
extern template class LoopBase<aiir::Block, aiir::CFGLoop>;
extern template class LoopInfoBase<aiir::Block, aiir::CFGLoop>;
} // namespace llvm

namespace aiir {

/// Representation of a single loop formed by blocks. The inherited LoopBase
/// class provides accessors to the loop analysis.
class CFGLoop : public llvm::LoopBase<aiir::Block, aiir::CFGLoop> {
private:
  explicit CFGLoop(aiir::Block *block);

  friend class llvm::LoopBase<aiir::Block, CFGLoop>;
  friend class llvm::LoopInfoBase<aiir::Block, CFGLoop>;
};

/// An LLVM LoopInfo instantiation for AIIR that provides access to CFG loops
/// found in the dominator tree.
class CFGLoopInfo : public llvm::LoopInfoBase<aiir::Block, aiir::CFGLoop> {
public:
  CFGLoopInfo(const llvm::DominatorTreeBase<aiir::Block, false> &domTree);
};

} // namespace aiir

#endif // AIIR_ANALYSIS_LOOPINFO_H
