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

#include <iterator>

using namespace llvm;
using namespace llvm::bolt;

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

raw_ostream &MCInstReference::print(raw_ostream &OS) const {
  if (const RefInBB *Ref = tryGetRefInBB()) {
    OS << "MCInstBBRef<";
    if (Ref->BB == nullptr) {
      OS << "BB:(null)";
    } else {
      unsigned IndexInBB = std::distance(Ref->BB->begin(), Ref->It);
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
