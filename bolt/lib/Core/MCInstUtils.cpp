//===- bolt/Core/MCInstUtils.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/MCInstUtils.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"

#include <type_traits>

using namespace llvm;
using namespace llvm::bolt;

// It is assumed in a few places that BinaryBasicBlock stores its instructions
// in a contiguous vector.
using BasicBlockStorageIsVector =
    std::is_same<BinaryBasicBlock::const_iterator,
                 std::vector<MCInst>::const_iterator>;
static_assert(BasicBlockStorageIsVector::value);

MCInstReference MCInstReference::get(const MCInst &Inst,
                                     const BinaryFunction &BF) {
  if (BF.hasCFG()) {
    for (BinaryBasicBlock &BB : BF) {
      for (MCInst &MI : BB)
        if (&MI == &Inst)
          return MCInstReference(BB, Inst);
    }
    llvm_unreachable("Inst is not contained in BF");
  }

  for (auto I = BF.instrs().begin(), E = BF.instrs().end(); I != E; ++I) {
    if (&I->second == &Inst)
      return MCInstReference(BF, I);
  }
  llvm_unreachable("Inst is not contained in BF");
}

uint64_t MCInstReference::computeAddress(const MCCodeEmitter *Emitter) const {
  assert(!empty() && "Taking instruction address by empty reference");

  const BinaryContext &BC = getFunction()->getBinaryContext();
  if (auto *Ref = tryGetRefInBB()) {
    const uint64_t AddressOfBB =
        getFunction()->getAddress() + Ref->BB->getOffset();
    const MCInst *FirstInstInBB = &*Ref->BB->begin();
    const MCInst *ThisInst = &getMCInst();

    // Usage of plain 'const MCInst *' as iterators assumes the instructions
    // are stored in a vector, see BasicBlockStorageIsVector.
    const uint64_t OffsetInBB =
        BC.computeCodeSize(FirstInstInBB, ThisInst, Emitter);

    return AddressOfBB + OffsetInBB;
  }

  auto &Ref = getRefInBF();
  const uint64_t OffsetInBF = Ref.It->first;

  return getFunction()->getAddress() + OffsetInBF;
}

raw_ostream &MCInstReference::print(raw_ostream &OS) const {
  if (const RefInBB *Ref = tryGetRefInBB()) {
    OS << "MCInstBBRef<";
    if (Ref->BB == nullptr)
      OS << "BB:(null)";
    else
      OS << "BB:" << Ref->BB->getName() << ":" << Ref->Index;
    OS << ">";
    return OS;
  }

  const RefInBF &Ref = getRefInBF();
  OS << "MCInstBFRef<";
  if (Ref.BF == nullptr)
    OS << "BF:(null)";
  else
    OS << "BF:" << Ref.BF->getPrintName() << ":" << Ref.It->first;
  OS << ">";
  return OS;
}
