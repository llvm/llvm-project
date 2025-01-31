//===- bolt/Passes/ReorderSection.cpp - Reordering of section data --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements DiscoverNoReturnPass class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/DiscoverNoReturnPass.h"

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltCategory;

static cl::opt<bool> DiscoverNoReturnAnalysis(
    "discover-no-return",
    cl::desc("analyze the binary and mark no-return functions"), cl::init(true),
    cl::cat(BoltCategory), cl::ReallyHidden);
} // namespace opts

namespace llvm {
namespace bolt {

Error DiscoverNoReturnPass::runOnFunctions(BinaryContext &BC) {
  bool Changed;
  do {
    Changed = false;
    for (auto &BFI : BC.getBinaryFunctions()) {
      auto &Func = BFI.second;
      bool PrevStat = BC.hasPathToNoReturn(&Func);
      bool CurStat = traverseFromFunction(&Func, BC);
      Changed |= (PrevStat != CurStat);
    }
  } while (Changed);

  return Error::success();
}

bool DiscoverNoReturnPass::traverseFromFunction(BinaryFunction *Func,
                                                BinaryContext &BC) {
  // The Function cached before, so return its value
  if (BC.cachedInNoReturnMap(Func))
    return BC.hasPathToNoReturn(Func);

  Visited[Func] = true;
  bool Result = true;
  bool hasCalls = 0;
  bool hasReturns = 0;
  for (auto &BB : *Func) {
    if (!BB.getNumCalls())
      continue;
    for (auto &Inst : BB) {
      if (BC.MIB->isCall(Inst)) {
        hasCalls = true;
        const MCSymbol *TargetSymbol = BC.MIB->getTargetSymbol(Inst);
        BinaryFunction *TargetFunction = BC.SymbolToFunctionMap[TargetSymbol];
        if (!Visited.count(TargetFunction))
          Result &= traverseFromFunction(TargetFunction, BC);
      }
      hasReturns |= BC.MIB->isReturn(Inst);
    }
  }

  // This functions is represented as a leaf in the call graph and doesn't
  // have a no-return attribute.
  if (!hasCalls && hasReturns)
    Result = false;

  // If the function doens't have a return instruction then it's a
  // no-return function.
  if (!hasReturns)
    Result = true;

  BC.setHasPathToNoReturn(Func, Result);
  return Result;
}

} // end namespace bolt
} // end namespace llvm
