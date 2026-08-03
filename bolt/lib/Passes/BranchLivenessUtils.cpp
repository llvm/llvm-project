//===- bolt/Passes/BranchLivenessUtils.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/BranchLivenessUtils.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Passes/DataflowInfoManager.h"
#include "bolt/Passes/RegAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCRegister.h"

namespace llvm {
namespace bolt {

bool hasShortRangeBranch(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();
  if (!BC.isAArch64())
    return false;

  return llvm::any_of(BF, [&](BinaryBasicBlock &BB) {
    return llvm::any_of(
        BB, [&](MCInst &Inst) { return BC.MIB->isShortRangeBranch(Inst); });
  });
}

DenseSet<const MCInst *> computeDeadFlagBranches(BinaryFunction &BF,
                                                 RegAnalysis &RA) {
  BinaryContext &BC = BF.getBinaryContext();
  SmallVector<MCInst *, 4> Insts;
  if (BC.isAArch64())
    for (BinaryBasicBlock &BB : BF)
      for (MCInst &Inst : BB)
        if (BC.MIB->isShortRangeBranch(Inst))
          Insts.push_back(&Inst);

  DenseSet<const MCInst *> DeadFlagBranches;
  if (Insts.empty())
    return DeadFlagBranches;

  DataflowInfoManager DIM(BF, &RA, nullptr);
  LivenessAnalysis &LA = DIM.getLivenessAnalysis();
  const MCPhysReg FlagsReg = BC.MIB->getFlagsReg();
  for (MCInst *Inst : Insts)
    if (!LA.getLiveIn(*Inst).test(FlagsReg))
      DeadFlagBranches.insert(Inst);
  return DeadFlagBranches;
}

} // namespace bolt
} // namespace llvm
