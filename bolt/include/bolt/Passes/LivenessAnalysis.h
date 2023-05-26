//===- bolt/Passes/LivenessAnalysis.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_LIVENESSANALYSIS_H
#define BOLT_PASSES_LIVENESSANALYSIS_H

#include "bolt/Passes/DataflowAnalysis.h"
#include "bolt/Passes/RegAnalysis.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/CommandLine.h"

namespace opts {
extern llvm::cl::opt<bool> AssumeABI;
extern llvm::cl::opt<bool> TimeOpts;
} // namespace opts

namespace llvm {
namespace bolt {

class LivenessAnalysis : public DataflowAnalysis<LivenessAnalysis, BitVector,
                                                 true, RegStatePrinter> {
  using Parent =
      DataflowAnalysis<LivenessAnalysis, BitVector, true, RegStatePrinter>;
  friend class DataflowAnalysis<LivenessAnalysis, BitVector, true,
                                RegStatePrinter>;

public:
  LivenessAnalysis(const RegAnalysis &RA, BinaryFunction &BF,
                   MCPlusBuilder::AllocatorIdTy AllocId)
      : Parent(BF, AllocId), RA(RA),
        NumRegs(BF.getBinaryContext().MRI->getNumRegs()) {}
  virtual ~LivenessAnalysis();

  bool isAlive(ProgramPoint PP, MCPhysReg Reg) const {
    BitVector BV = (*this->getStateAt(PP));
    const BitVector &RegAliases = BC.MIB->getAliases(Reg);
    BV &= RegAliases;
    return BV.any();
  }

  void run() { Parent::run(); }

  // Return a usable general-purpose reg after point P. Return 0 if no reg is
  // available.
  MCPhysReg scavengeRegAfter(ProgramPoint P) {
    BitVector BV = *this->getStateAt(P);
    BV.flip();
    BitVector GPRegs(NumRegs, false);
    this->BC.MIB->getGPRegs(GPRegs, /*IncludeAlias=*/false);
    // Ignore the register used for frame pointer even if it is not alive (it
    // may be used by CFI which is not represented in our dataflow).
    BitVector FP = BC.MIB->getAliases(BC.MIB->getFramePointer());
    FP.flip();
    BV &= GPRegs;
    BV &= FP;
    int Reg = BV.find_first();
    return Reg != -1 ? Reg : 0;
  }

protected:
  /// Reference to the result of reg analysis
  const RegAnalysis &RA;
  const uint16_t NumRegs;

  void preflight() {}

  BitVector getStartingStateAtBB(const BinaryBasicBlock &BB) {
    // Entry points start with default live out (registers used as return
    // values).
    if (BB.succ_size() == 0) {
      BitVector State(NumRegs, false);
      if (opts::AssumeABI) {
        BC.MIB->getDefaultLiveOut(State);
        BC.MIB->getCalleeSavedRegs(State);
      } else {
        State.set();
        State.reset(BC.MIB->getFlagsReg());
      }
      return State;
    }
    return BitVector(NumRegs, false);
  }

  BitVector getStartingStateAtPoint(const MCInst &Point) {
    return BitVector(NumRegs, false);
  }

  void doConfluence(BitVector &StateOut, const BitVector &StateIn) {
    StateOut |= StateIn;
  }

  BitVector computeNext(const MCInst &Point, const BitVector &Cur) {
    BitVector Next = Cur;
    bool IsCall = this->BC.MIB->isCall(Point);
    // Kill
    BitVector Written = BitVector(NumRegs, false);
    if (!IsCall) {
      this->BC.MIB->getWrittenRegs(Point, Written);
    } else {
      RA.getInstClobberList(Point, Written);
      // When clobber list is conservative, it is clobbering all/most registers,
      // a conservative estimate because it knows nothing about this call.
      // For our purposes, assume it kills no registers/callee-saved regs
      // because we don't really know what's going on.
      if (RA.isConservative(Written)) {
        Written.reset();
        BC.MIB->getDefaultLiveOut(Written);
        // If ABI is respected, everything except CSRs should be dead after a
        // call
        if (opts::AssumeABI) {
          BitVector CSR = BitVector(NumRegs, false);
          BC.MIB->getCalleeSavedRegs(CSR);
          CSR.flip();
          Written |= CSR;
        }
      }
    }
    Written.flip();
    Next &= Written;
    // Gen
    if (!this->BC.MIB->isCFI(Point)) {
      if (BC.MIB->isCleanRegXOR(Point))
        return Next;

      BitVector Used = BitVector(NumRegs, false);
      if (IsCall) {
        RA.getInstUsedRegsList(Point, Used, /*GetClobbers*/ true);
        if (RA.isConservative(Used)) {
          Used = BC.MIB->getRegsUsedAsParams();
          BC.MIB->getDefaultLiveOut(Used);
        }
      }
      const MCInstrDesc &InstInfo = BC.MII->get(Point.getOpcode());
      for (const MCOperand &Op : BC.MIB->useOperands(Point))
        if (Op.isReg())
          Used |= BC.MIB->getAliases(Op.getReg(), /*OnlySmaller=*/false);
      for (MCPhysReg ImplicitUse : InstInfo.implicit_uses())
        Used |= BC.MIB->getAliases(ImplicitUse, false);
      if (IsCall &&
          (!BC.MIB->isTailCall(Point) || !BC.MIB->isConditionalBranch(Point))) {
        // Never gen FLAGS from a non-conditional call... this is overly
        // conservative
        Used.reset(BC.MIB->getFlagsReg());
      }
      Next |= Used;
    }
    return Next;
  }

  StringRef getAnnotationName() const { return StringRef("LivenessAnalysis"); }
};

} // end namespace bolt
} // end namespace llvm

#endif
