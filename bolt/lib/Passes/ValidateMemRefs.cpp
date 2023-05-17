//===- bolt/Passes/ValidateMemRefs.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/ValidateMemRefs.h"
#include "bolt/Core/ParallelUtilities.h"

#define DEBUG_TYPE "bolt-memrefs"

namespace opts {
extern llvm::cl::opt<llvm::bolt::JumpTableSupportLevel> JumpTables;
}

namespace llvm::bolt {

std::atomic<std::uint64_t> ValidateMemRefs::ReplacedReferences{0};

bool ValidateMemRefs::checkAndFixJTReference(BinaryFunction &BF, MCInst &Inst,
                                             uint32_t OperandNum,
                                             const MCSymbol *Sym,
                                             uint64_t Offset) {
  BinaryContext &BC = BF.getBinaryContext();
  auto L = BC.scopeLock();
  BinaryData *BD = BC.getBinaryDataByName(Sym->getName());
  if (!BD)
    return false;

  const uint64_t TargetAddress = BD->getAddress() + Offset;
  JumpTable *JT = BC.getJumpTableContainingAddress(TargetAddress);
  if (!JT)
    return false;

  const bool IsLegitAccess = llvm::any_of(
      JT->Parents, [&](const BinaryFunction *Parent) { return Parent == &BF; });
  if (IsLegitAccess)
    return true;

  // Accessing a jump table in another function. This is not a
  // legitimate jump table access, we need to replace the reference to
  // the jump table label with a regular rodata reference. Get a
  // non-JT reference by fetching the symbol 1 byte before the JT
  // label.
  MCSymbol *NewSym = BC.getOrCreateGlobalSymbol(TargetAddress - 1, "DATAat");
  BC.MIB->setOperandToSymbolRef(Inst, OperandNum, NewSym, 1, &*BC.Ctx, 0);
  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: replaced reference @" << BF.getPrintName()
                    << " from " << BD->getName() << " to " << NewSym->getName()
                    << " + 1\n");
  ++ReplacedReferences;
  return true;
}

void ValidateMemRefs::runOnFunction(BinaryFunction &BF) {
  MCPlusBuilder *MIB = BF.getBinaryContext().MIB.get();

  for (BinaryBasicBlock &BB : BF) {
    for (MCInst &Inst : BB) {
      for (int I = 0, E = MCPlus::getNumPrimeOperands(Inst); I != E; ++I) {
        const MCOperand &Operand = Inst.getOperand(I);
        if (!Operand.isExpr())
          continue;

        const auto [Sym, Offset] = MIB->getTargetSymbolInfo(Operand.getExpr());
        if (!Sym)
          continue;

        checkAndFixJTReference(BF, Inst, I, Sym, Offset);
      }
    }
  }
}

void ValidateMemRefs::runOnFunctions(BinaryContext &BC) {
  if (!BC.isX86())
    return;

  // Skip validation if not moving JT
  if (opts::JumpTables == JTS_NONE || opts::JumpTables == JTS_BASIC)
    return;

  ParallelUtilities::WorkFuncWithAllocTy ProcessFunction =
      [&](BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocId) {
        runOnFunction(BF);
      };
  ParallelUtilities::PredicateTy SkipPredicate = [&](const BinaryFunction &BF) {
    return !BF.hasCFG();
  };
  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: starting memrefs validation pass\n");
  ParallelUtilities::runOnEachFunctionWithUniqueAllocId(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_LINEAR, ProcessFunction,
      SkipPredicate, "validate-mem-refs", /*ForceSequential=*/true);
  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: memrefs validation is concluded\n");

  if (!ReplacedReferences)
    return;

  outs() << "BOLT-INFO: validate-mem-refs updated " << ReplacedReferences
         << " object references\n";
}

} // namespace llvm::bolt
