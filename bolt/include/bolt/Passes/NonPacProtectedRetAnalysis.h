//===- bolt/Passes/NonPacProtectedRetAnalysis.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_NONPACPROTECTEDRETANALYSIS_H
#define BOLT_PASSES_NONPACPROTECTEDRETANALYSIS_H

#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Passes/BinaryPasses.h"
#include "llvm/ADT/SmallSet.h"

namespace llvm {
namespace bolt {

/// @brief  MCInstReference represents a reference to an MCInst as stored either
/// in a BinaryFunction (i.e. before a CFG is created), or in a BinaryBasicBlock
/// (after a CFG is created). It aims to store the necessary information to be
/// able to find the specific MCInst in either the BinaryFunction or
/// BinaryBasicBlock data structures later, so that e.g. the InputAddress of
/// the corresponding instruction can be computed.

struct MCInstInBBReference {
  BinaryBasicBlock *BB;
  int64_t BBIndex;
  MCInstInBBReference(BinaryBasicBlock *BB, int64_t BBIndex)
      : BB(BB), BBIndex(BBIndex) {}
  MCInstInBBReference() : BB(nullptr), BBIndex(0) {}
  static MCInstInBBReference get(const MCInst *Inst, BinaryFunction &BF) {
    for (BinaryBasicBlock& BB : BF)
      for (size_t I = 0; I < BB.size(); ++I)
        if (Inst == &(BB.getInstructionAtIndex(I)))
          return MCInstInBBReference(&BB, I);
    return {};
  }
  bool operator==(const MCInstInBBReference &RHS) const {
    return BB == RHS.BB && BBIndex == RHS.BBIndex;
  }
  bool operator<(const MCInstInBBReference &RHS) const {
    if (BB != RHS.BB)
      return BB < RHS.BB;
    return BBIndex < RHS.BBIndex;
  }
  operator MCInst &() const {
    assert(BB != nullptr);
    return BB->getInstructionAtIndex(BBIndex);
  }
  uint64_t getAddress() const {
    // 4 bytes per instruction on AArch64;
    return BB->getFunction()->getAddress() + BB->getOffset() + BBIndex * 4;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const MCInstInBBReference &);

struct MCInstInBFReference {
  BinaryFunction *BF;
  uint32_t Offset;
  MCInstInBFReference(BinaryFunction *BF, uint32_t Offset)
      : BF(BF), Offset(Offset) {}
  MCInstInBFReference() : BF(nullptr) {}
  bool operator==(const MCInstInBFReference &RHS) const {
    return BF == RHS.BF && Offset == RHS.Offset;
  }
  bool operator<(const MCInstInBFReference &RHS) const {
    if (BF != RHS.BF)
      return BF < RHS.BF;
    return Offset < RHS.Offset;
  }
  operator MCInst &() const {
    assert(BF != nullptr);
    return *(BF->getInstructionAtOffset(Offset));
  }

  uint64_t getOffset() const { return Offset; }

  uint64_t getAddress() const {
    return BF->getAddress() + getOffset();
  }
};

raw_ostream &operator<<(raw_ostream &OS, const MCInstInBFReference &);

struct MCInstReference {
  enum StoredIn { _BinaryFunction, _BinaryBasicBlock };
  StoredIn CurrentLocation;
  union U {
    MCInstInBBReference BBRef;
    MCInstInBFReference BFRef;
    U(MCInstInBBReference BBRef) : BBRef(BBRef) {}
    U(MCInstInBFReference BFRef) : BFRef(BFRef) {}
  } U;
  MCInstReference(MCInstInBBReference BBRef)
      : CurrentLocation(_BinaryBasicBlock), U(BBRef) {}
  MCInstReference(MCInstInBFReference BFRef)
      : CurrentLocation(_BinaryFunction), U(BFRef) {}
  MCInstReference(BinaryBasicBlock *BB, int64_t BBIndex)
      : MCInstReference(MCInstInBBReference(BB, BBIndex)) {}
  MCInstReference(BinaryFunction *BF, uint32_t Offset)
      : MCInstReference(MCInstInBFReference(BF, Offset)) {}

  bool operator<(const MCInstReference &RHS) const {
    if (CurrentLocation != RHS.CurrentLocation)
      return CurrentLocation < RHS.CurrentLocation;
    switch (CurrentLocation) {
    case _BinaryBasicBlock:
      return U.BBRef < RHS.U.BBRef;
    case _BinaryFunction:
      return U.BFRef < RHS.U.BFRef;
    }
    llvm_unreachable("");
  }

  bool operator==(const MCInstReference &RHS) const {
    if (CurrentLocation != RHS.CurrentLocation)
      return false;
    switch (CurrentLocation) {
    case _BinaryBasicBlock:
      return U.BBRef == RHS.U.BBRef;
    case _BinaryFunction:
      return U.BFRef == RHS.U.BFRef;
    }
    llvm_unreachable("");
  }

  operator MCInst &() const {
    switch (CurrentLocation) {
    case _BinaryBasicBlock:
      return U.BBRef;
    case _BinaryFunction:
      return U.BFRef;
    }
    llvm_unreachable("");
  }

  uint64_t getAddress() const {
    switch (CurrentLocation) {
    case _BinaryBasicBlock:
      return U.BBRef.getAddress();
    case _BinaryFunction:
      return U.BFRef.getAddress();
    }
    llvm_unreachable("");
  }

  BinaryFunction *getFunction() const {
    switch (CurrentLocation) {
    case _BinaryFunction:
      return U.BFRef.BF;
    case _BinaryBasicBlock:
      return U.BBRef.BB->getFunction();
    }
    llvm_unreachable("");
  }

  BinaryBasicBlock *getBasicBlock() const {
    switch (CurrentLocation) {
    case _BinaryFunction:
      return nullptr;
    case _BinaryBasicBlock:
      return U.BBRef.BB;
    }
    llvm_unreachable("");
  }
};

raw_ostream &operator<<(raw_ostream &OS, const MCInstReference &);

struct NonPacProtectedRetGadget {
  MCInstReference RetInst;
  std::vector<MCInstReference> OverwritingRetRegInst;
  bool operator==(const NonPacProtectedRetGadget &RHS) const {
    return RetInst == RHS.RetInst &&
           OverwritingRetRegInst == RHS.OverwritingRetRegInst;
  }
  NonPacProtectedRetGadget(
      MCInstReference RetInst,
      const std::vector<MCInstReference>& OverwritingRetRegInst)
      : RetInst(RetInst), OverwritingRetRegInst(OverwritingRetRegInst) {}
};

raw_ostream &operator<<(raw_ostream &OS, const NonPacProtectedRetGadget &NPPRG);
class PacRetAnalysis;

class NonPacProtectedRetAnalysis : public BinaryFunctionPass {
  void runOnFunction(BinaryFunction &Function,
                     MCPlusBuilder::AllocatorIdTy AllocatorId);
  SmallSet<MCPhysReg, 1>
  computeDfState(PacRetAnalysis &PRA, BinaryFunction &BF,
                 MCPlusBuilder::AllocatorIdTy AllocatorId);
  unsigned GadgetAnnotationIndex;

public:
  explicit NonPacProtectedRetAnalysis() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "non-pac-protected-rets"; }

  /// Pass entry point
  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif