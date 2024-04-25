//===- bolt/Passes/FixRISCVCallsPass.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/FixRISCVCallsPass.h"
#include "bolt/Core/ParallelUtilities.h"

#include <iterator>

using namespace llvm;

namespace llvm {
namespace bolt {

void FixRISCVCallsPass::runOnFunction(BinaryFunction &BF) {
  auto &BC = BF.getBinaryContext();
  auto &MIB = BC.MIB;
  auto *Ctx = BC.Ctx.get();

  for (auto &BB : BF) {
    for (auto II = BB.begin(); II != BB.end();) {
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
        ++II;
        continue;
      }

      auto NextII = std::next(II);

      if (NextII == BB.end())
        break;

      if (MIB->isRISCVCall(*II, *NextII)) {
        auto *Target = MIB->getTargetSymbol(*II);
        assert(Target && "Cannot find call target");

        MCInst OldCall = *NextII;
        auto L = BC.scopeLock();

        if (MIB->isTailCall(*NextII))
          MIB->createTailCall(*II, Target, Ctx);
        else
          MIB->createCall(*II, Target, Ctx);

        MIB->moveAnnotations(std::move(OldCall), *II);

        // The original offset was set on the jalr of the auipc+jalr pair. Since
        // the whole pair is replaced by a call, adjust the offset by -4 (the
        // size of a auipc).
        if (std::optional<uint32_t> Offset = MIB->getOffset(*II)) {
          assert(*Offset >= 4 && "Illegal jalr offset");
          MIB->setOffset(*II, *Offset - 4);
        }

        II = BB.eraseInstruction(NextII);
        continue;
      }

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
