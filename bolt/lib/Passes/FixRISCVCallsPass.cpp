//===- bolt/Passes/FixRISCVCallsPass.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/FixRISCVCallsPass.h"
#include "bolt/Core/ParallelUtilities.h"

using namespace llvm;

namespace llvm {
namespace bolt {

void FixRISCVCallsPass::runOnFunction(BinaryFunction &BF) {
  auto &BC = BF.getBinaryContext();
  auto &MIB = BC.MIB;
  auto *Ctx = BC.Ctx.get();

  MCInst *Previous = nullptr;
  BinaryBasicBlock *PreviousBB = nullptr;
  for (auto &BB : BF) {
    for (auto II = BB.begin(); II != BB.end();) {
      // CFI and other zero-sized pseudo instructions do not break an
      // AUIPC/JALR pair in the input instruction stream.
      if (MIB->isPseudo(*II)) {
        ++II;
        continue;
      }

      if (MIB->isCall(*II) && !MIB->isIndirectCall(*II)) {
        auto *Target = MIB->getTargetSymbol(*II);
        assert(Target && "Cannot find call target");

        MCInst OldCall = *II;
        auto L = BC.scopeLock();

        if (MIB->isTailCall(*II))
          MIB->createTailCall(*II, Target, Ctx);
        else
          MIB->createCall(*II, Target, Ctx);

        MIB->moveAnnotations(std::move(OldCall), *II);
        Previous = &*II;
        PreviousBB = &BB;
        ++II;
        continue;
      }

      // A label, secondary entry point, or CFG boundary may split an
      // AUIPC/JALR call pair across two basic blocks. Keep the previous real
      // instruction across block boundaries so that the pair is still
      // rewritten atomically. Otherwise the old JALR immediate remains in the
      // encoding and can be ORed with the new R_RISCV_CALL_PLT fixup.
      if (Previous && MIB->isRISCVCall(*Previous, *II)) {
        auto *Target = MIB->getTargetSymbol(*Previous);
        assert(Target && "Cannot find call target");

        MCInst OldCall = *II;
        auto L = BC.scopeLock();

        if (PreviousBB == &BB) {
          // Keep the original JALR offset annotation on the combined call, but
          // emit the pseudo at the AUIPC position and remove the JALR. This
          // preserves profile attribution without adding an executed NOP to
          // every long call.
          if (MIB->isTailCall(*II))
            MIB->createTailCall(*Previous, Target, Ctx);
          else
            MIB->createCall(*Previous, Target, Ctx);
          MIB->moveAnnotations(std::move(OldCall), *Previous);
          II = BB.eraseInstruction(II);
        } else {
          // Keep split pairs in their original basic blocks. Moving the call
          // across a CFG boundary would invalidate block-level control-flow
          // information.
          MIB->createNoop(*Previous);
          if (MIB->isTailCall(*II))
            MIB->createTailCall(*II, Target, Ctx);
          else
            MIB->createCall(*II, Target, Ctx);
          MIB->moveAnnotations(std::move(OldCall), *II);
          ++II;
        }
        Previous = nullptr;
        PreviousBB = nullptr;
        continue;
      }

      Previous = &*II;
      PreviousBB = &BB;
      ++II;
    }
  }
}

Error FixRISCVCallsPass::runOnFunctions(BinaryContext &BC) {
  if (!BC.isRISCV() || !BC.HasRelocations)
    return Error::success();

  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    runOnFunction(BF);
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_LINEAR, WorkFun, nullptr,
      "FixRISCVCalls");

  return Error::success();
}

} // namespace bolt
} // namespace llvm
