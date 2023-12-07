#include "bolt/Passes/FixRISCVCallsPass.h"
#include "bolt/Core/ParallelUtilities.h"

#include <iterator>

using namespace llvm;

namespace llvm {
namespace bolt {

void FixRISCVCallsPass::runOnFunction(BinaryFunction &BF) {
  auto &BC = BF.getBinaryContext();

  for (auto &BB : BF) {
    for (auto II = BB.begin(), IE = BB.end(); II != IE; ++II) {
      auto NextII = std::next(II);

      if (NextII == IE)
        break;

      if (!BC.MIB->isRISCVCall(*II, *NextII))
        continue;

      auto L = BC.scopeLock();

      // The MC layer handles R_RISCV_CALL_PLT but assumes that the immediate
      // in the JALR is zero (fixups are or'ed into instructions). Note that
      // NextII is guaranteed to point to a JALR by isRISCVCall.
      NextII->getOperand(2).setImm(0);
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
