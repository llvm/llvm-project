//===- bolt/Passes/MCInstUtils.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/MCInstUtils.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/ADT/iterator.h"

#include <type_traits>

using namespace llvm;
using namespace llvm::bolt;

// It is assumed in a few places that BinaryBasicBlock stores its instructions
// in a contiguous vector. Give this assumption a name to simplify marking the
// particular places with static_assert.
using BasicBlockStorageIsVector =
    std::is_same<BinaryBasicBlock::const_iterator,
                 std::vector<MCInst>::const_iterator>;

namespace {
// Cannot reuse MCPlusBuilder::InstructionIterator because it has to be
// constructed from a non-const std::map iterator.
class mapped_mcinst_iterator
    : public iterator_adaptor_base<mapped_mcinst_iterator,
                                   MCInstReference::nocfg_const_iterator> {
public:
  mapped_mcinst_iterator(MCInstReference::nocfg_const_iterator It)
      : iterator_adaptor_base(It) {}
  const MCInst &operator*() const { return this->I->second; }
};
} // anonymous namespace

MCInstReference MCInstReference::get(const MCInst *Inst,
                                     const BinaryFunction &BF) {
  if (BF.hasCFG()) {
    for (BinaryBasicBlock &BB : BF)
      for (MCInst &MI : BB)
        if (&MI == Inst)
          return MCInstReference(&BB, Inst);
    return {};
  }

  for (auto I = BF.instrs().begin(), E = BF.instrs().end(); I != E; ++I) {
    if (&I->second == Inst)
      return MCInstReference(&BF, I);
  }
  return {};
}

uint64_t MCInstReference::getAddress(const MCCodeEmitter *Emitter) const {
  assert(!empty() && "Taking instruction address by empty reference");

  const BinaryContext &BC = getFunction()->getBinaryContext();
  if (auto *Ref = tryGetRefInBB()) {
    static_assert(BasicBlockStorageIsVector::value,
                  "Cannot use 'const MCInst *' as iterator type");
    uint64_t AddressOfBB = getFunction()->getAddress() + Ref->BB->getOffset();
    const MCInst *FirstInstInBB = &*Ref->BB->begin();

    uint64_t OffsetInBB = BC.computeCodeSize(FirstInstInBB, Ref->Inst, Emitter);

    return AddressOfBB + OffsetInBB;
  }

  auto &Ref = getRefInBF();
  mapped_mcinst_iterator FirstInstInBF(Ref.BF->instrs().begin());
  mapped_mcinst_iterator ThisInst(Ref.It);

  uint64_t OffsetInBF = BC.computeCodeSize(FirstInstInBF, ThisInst, Emitter);

  return getFunction()->getAddress() + OffsetInBF;
}

raw_ostream &MCInstReference::print(raw_ostream &OS) const {
  if (const RefInBB *Ref = tryGetRefInBB()) {
    OS << "MCInstBBRef<";
    if (Ref->BB == nullptr) {
      OS << "BB:(null)";
    } else {
      static_assert(BasicBlockStorageIsVector::value,
                    "Cannot use pointer arithmetic on 'const MCInst *'");
      const MCInst *FirstInstInBB = &*Ref->BB->begin();
      unsigned IndexInBB = Ref->Inst - FirstInstInBB;
      OS << "BB:" << Ref->BB->getName() << ":" << IndexInBB;
    }
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
