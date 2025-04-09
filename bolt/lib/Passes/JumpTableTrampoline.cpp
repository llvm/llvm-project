//===- bolt/Passes/JumpTableTrampoline.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements JumpTableTrampoline class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/JumpTableTrampoline.h"

#define DEBUG_TYPE "JTT"

namespace llvm {
namespace bolt {

void JumpTableTrampoline::optimizeFunction(BinaryFunction &Function) {
  BinaryContext &BC = Function.getBinaryContext();
  std::vector<std::unique_ptr<BinaryBasicBlock>> NewBBs;
  InstructionListType Seq;
  for (JumpTable *const &JT : llvm::make_second_range(Function.jumpTables())) {
    for (MCSymbol *&Entry : JT->Entries) {
      BinaryBasicBlock *Target = Function.getBasicBlockForLabel(Entry);
      assert(Target && "Jump table entry must have corresponding basic block");
      if (!Target->isCold())
        continue;

      std::unique_ptr<BinaryBasicBlock> NewBB =
          Function.createBasicBlock(BC.Ctx->createNamedTempSymbol("JTT"));
      NewBBs.emplace_back(std::move(NewBB));

      // Copy attributes from the original destination.
      BinaryBasicBlock &NewBBRef = *NewBBs.back();
      NewBBRef.setOffset(Target->getOffset());

      // Populate the new basic block.
      BC.MIB->createLongJmp(Seq, Entry, BC.Ctx.get());
      NewBBRef.addInstructions(Seq);
      Seq.clear();

      // Redirect the jump table entry.
      Entry = NewBBRef.getLabel();
    }
  }

  if (NewBBs.empty())
    return;

  Modified.insert(&Function);
  BinaryBasicBlock *LastHotBB = Function.getLayout().getMainFragment().back();
  assert(LastHotBB && "split function must have at least one hot block");
  Function.insertBasicBlocks(LastHotBB, std::move(NewBBs));
}

Error JumpTableTrampoline::runOnFunctions(BinaryContext &BC) {
  for (BinaryFunction &BF : llvm::make_second_range(BC.getBinaryFunctions()))
    if (BF.isSplit() && BF.hasJumpTables())
      optimizeFunction(BF);

  BC.outs() << "BOLT-INFO: inserted jump table trampolines into "
            << Modified.size() << " functions.\n";
  return Error::success();
}

} // namespace bolt
} // namespace llvm
