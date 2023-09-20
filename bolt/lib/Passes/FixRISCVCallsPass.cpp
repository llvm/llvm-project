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

        auto L = BC.scopeLock();

        if (MIB->isTailCall(*II))
          MIB->createTailCall(*II, Target, Ctx);
        else
          MIB->createCall(*II, Target, Ctx);

        ++II;
        continue;
      }

      auto NextII = std::next(II);

      if (NextII == BB.end())
        break;

      if (MIB->isRISCVCall(*II, *NextII)) {
        auto *Target = MIB->getTargetSymbol(*II);
        assert(Target && "Cannot find call target");

        auto L = BC.scopeLock();
        MIB->createCall(*II, Target, Ctx);
        II = BB.eraseInstruction(NextII);
        continue;
      }

      ++II;
    }
  }
}

void FixRISCVCallsPass::runOnFunctions(BinaryContext &BC) {
  if (!BC.isRISCV() || !BC.HasRelocations)
    return;

  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    runOnFunction(BF);
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_LINEAR, WorkFun, nullptr,
      "FixRISCVCalls");
}

} // namespace bolt
} // namespace llvm
