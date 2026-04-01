//===- CFGLoopInfo.cpp - LoopInfo analysis for region bodies --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Analysis/CFGLoopInfo.h"
#include "llvm/Support/GenericLoopInfoImpl.h"

// Explicitly instantiate the LoopBase and LoopInfoBase classes defined in
// LoopInfoImpl.h for CFGLoops
template class llvm::LoopBase<aiir::Block, aiir::CFGLoop>;
template class llvm::LoopInfoBase<aiir::Block, aiir::CFGLoop>;

using namespace aiir;

CFGLoop::CFGLoop(aiir::Block *block)
    : llvm::LoopBase<aiir::Block, CFGLoop>(block) {}

CFGLoopInfo::CFGLoopInfo(
    const llvm::DominatorTreeBase<aiir::Block, false> &domTree) {
  analyze(domTree);
}
