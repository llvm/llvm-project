#include "bolt/Passes/FixRelaxationPass.h"
#include "bolt/Core/ParallelUtilities.h"

using namespace llvm;

namespace llvm {
namespace bolt {

// This function finds ADRP+ADD instruction sequences that originally before
// linker relaxations were ADRP+LDR. We've modified LDR/ADD relocation properly
// during relocation reading, so its targeting right symbol. As for ADRP its
// target is wrong before this pass since we won't be able to recognize and
// properly change R_AARCH64_ADR_GOT_PAGE relocation to
// R_AARCH64_ADR_PREL_PG_HI21 during relocation reading. Now we're searching for
// ADRP+ADD sequences, checking that ADRP points to the GOT-table symbol and the
// target of ADD is another symbol. When found change ADRP symbol reference to
// the ADDs one.
void FixRelaxations::runOnFunction(BinaryFunction &BF) {
  BinaryContext &BC = BF.getBinaryContext();
  for (BinaryBasicBlock &BB : BF) {
    for (auto II = BB.begin(); II != BB.end(); ++II) {
      MCInst &Adrp = *II;
      if (BC.MIB->isPseudo(Adrp) || !BC.MIB->isADRP(Adrp))
        continue;

      const MCSymbol *AdrpSymbol = BC.MIB->getTargetSymbol(Adrp);
      if (!AdrpSymbol || AdrpSymbol->getName() != "__BOLT_got_zero")
        continue;

      auto NextII = std::next(II);
      if (NextII == BB.end())
        continue;

      const MCInst &Add = *NextII;
      if (!BC.MIB->matchAdrpAddPair(Adrp, Add))
        continue;

      const MCSymbol *Symbol = BC.MIB->getTargetSymbol(Add);
      if (!Symbol || AdrpSymbol == Symbol)
        continue;

      auto L = BC.scopeLock();
      const int64_t Addend = BC.MIB->getTargetAddend(Add);
      BC.MIB->setOperandToSymbolRef(Adrp, /*OpNum*/ 1, Symbol, Addend,
                                    BC.Ctx.get(), ELF::R_AARCH64_NONE);
    }
  }
}

Error FixRelaxations::runOnFunctions(BinaryContext &BC) {
  if (!BC.isAArch64() || !BC.HasRelocations)
    return Error::success();

  ParallelUtilities::WorkFuncTy WorkFun = [&](BinaryFunction &BF) {
    runOnFunction(BF);
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_LINEAR, WorkFun, nullptr,
      "FixRelaxations");
  return Error::success();
}

} // namespace bolt
} // namespace llvm
