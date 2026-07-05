//===- bolt/Passes/StackPointerTracking.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_STACKPOINTERTRACKING_H
#define BOLT_PASSES_STACKPOINTERTRACKING_H

#include "bolt/Passes/DataflowAnalysis.h"
#include "llvm/Support/CommandLine.h"

namespace opts {
extern llvm::cl::opt<bool> TimeOpts;
} // namespace opts

namespace llvm {
namespace bolt {

/// Perform a dataflow analysis to track the value of SP as an offset relative
/// to the CFA.
template <typename Derived>
class StackPointerTrackingBase
    : public DataflowAnalysis<Derived, std::pair<int, int>> {
  friend class DataflowAnalysis<Derived, std::pair<int, int>>;

protected:
  void preflight() {}

  int getEmpty() { return EMPTY; }

  std::pair<int, int> getStartingStateAtBB(const BinaryBasicBlock &BB) {
    // Entry BB start with offset 8 from CFA.
    // All others start with EMPTY (meaning we don't know anything).
    if (BB.isEntryPoint())
      return std::make_pair(-8, getEmpty());
    return std::make_pair(getEmpty(), getEmpty());
  }

  std::pair<int, int> getStartingStateAtPoint(const MCInst &Point) {
    return std::make_pair(getEmpty(), getEmpty());
  }

  void doConfluenceSingleReg(int &StateOut, const int &StateIn) {
    if (StateOut == EMPTY) {
      StateOut = StateIn;
      return;
    }
    if (StateIn == EMPTY || StateIn == StateOut)
      return;

    // We can't agree on a specific value from this point on
    StateOut = SUPERPOSITION;
  }

  void doConfluence(std::pair<int, int> &StateOut,
                    const std::pair<int, int> &StateIn) {
    doConfluenceSingleReg(StateOut.first, StateIn.first);
    doConfluenceSingleReg(StateOut.second, StateIn.second);
  }

  void doConfluenceWithLP(std::pair<int, int> &StateOut,
                          const std::pair<int, int> &StateIn,
                          const MCInst &Invoke) {
    int SPVal = StateIn.first;
    if (SPVal != EMPTY && SPVal != SUPERPOSITION) {
      const int64_t GnuArgsSize = this->BC.MIB->getGnuArgsSize(Invoke);
      if (GnuArgsSize > 0)
        SPVal += GnuArgsSize;
    }
    doConfluenceSingleReg(StateOut.first, SPVal);
    doConfluenceSingleReg(StateOut.second, StateIn.second);
  }

  int computeNextSP(const MCInst &Point, int SPVal, int FPVal) {
    const auto &MIB = this->BC.MIB;

    if (int Sz = MIB->getPushSize(Point)) {
      if (SPVal == EMPTY || SPVal == SUPERPOSITION)
        return SPVal;

      return SPVal - Sz;
    }

    if (int Sz = MIB->getPopSize(Point)) {
      if (SPVal == EMPTY || SPVal == SUPERPOSITION)
        return SPVal;

      return SPVal + Sz;
    }

    MCPhysReg From, To;
    if (MIB->isRegToRegMove(Point, From, To) && To == MIB->getStackPointer() &&
        From == MIB->getFramePointer()) {
      if (FPVal == EMPTY || FPVal == SUPERPOSITION)
        return FPVal;

      if (MIB->isLeave(Point))
        return FPVal + 8;
      return FPVal;
    }

    if (this->BC.MII->get(Point.getOpcode())
            .hasDefOfPhysReg(Point, MIB->getStackPointer(), *this->BC.MRI)) {
      std::pair<MCPhysReg, int64_t> SP;
      if (SPVal != EMPTY && SPVal != SUPERPOSITION)
        SP = std::make_pair(MIB->getStackPointer(), SPVal);
      else
        SP = std::make_pair(0, 0);
      std::pair<MCPhysReg, int64_t> FP;
      if (FPVal != EMPTY && FPVal != SUPERPOSITION)
        FP = std::make_pair(MIB->getFramePointer(), FPVal);
      else
        FP = std::make_pair(0, 0);
      int64_t Output;
      if (!MIB->evaluateStackOffsetExpr(Point, Output, SP, FP)) {
        if (SPVal == EMPTY && FPVal == EMPTY)
          return SPVal;
        return SUPERPOSITION;
      }

      return static_cast<int>(Output);
    }

    return SPVal;
  }

  int computeNextFP(const MCInst &Point, int SPVal, int FPVal) {
    const auto &MIB = this->BC.MIB;

    MCPhysReg From, To;
    if (MIB->isRegToRegMove(Point, From, To) && To == MIB->getFramePointer() &&
        From == MIB->getStackPointer()) {
      HasFramePointer = true;
      return SPVal;
    }

    if (this->BC.MII->get(Point.getOpcode())
            .hasDefOfPhysReg(Point, MIB->getFramePointer(), *this->BC.MRI)) {
      std::pair<MCPhysReg, int64_t> FP;
      if (FPVal != EMPTY && FPVal != SUPERPOSITION)
        FP = std::make_pair(MIB->getFramePointer(), FPVal);
      else
        FP = std::make_pair(0, 0);
      std::pair<MCPhysReg, int64_t> SP;
      if (SPVal != EMPTY && SPVal != SUPERPOSITION)
        SP = std::make_pair(MIB->getStackPointer(), SPVal);
      else
        SP = std::make_pair(0, 0);
      int64_t Output;
      if (!MIB->evaluateStackOffsetExpr(Point, Output, SP, FP)) {
        if (SPVal == EMPTY && FPVal == EMPTY)
          return FPVal;
        return SUPERPOSITION;
      }

      if (!HasFramePointer && MIB->escapesVariable(Point, false))
        HasFramePointer = true;
      return static_cast<int>(Output);
    }

    return FPVal;
  }

  std::pair<int, int> computeNext(const MCInst &Point,
                                  const std::pair<int, int> &Cur) {
    return std::make_pair(computeNextSP(Point, Cur.first, Cur.second),
                          computeNextFP(Point, Cur.first, Cur.second));
  }

  StringRef getAnnotationName() const {
    return StringRef("StackPointerTracking");
  }

public:
  StackPointerTrackingBase(BinaryFunction &BF,
                           MCPlusBuilder::AllocatorIdTy AllocatorId = 0)
      : DataflowAnalysis<Derived, std::pair<int, int>>(BF, AllocatorId) {}

  virtual ~StackPointerTrackingBase() {}

  bool HasFramePointer{false};

  static constexpr int SUPERPOSITION = std::numeric_limits<int>::max();
  static constexpr int EMPTY = std::numeric_limits<int>::min();
};

class StackPointerTracking
    : public StackPointerTrackingBase<StackPointerTracking> {
  friend class DataflowAnalysis<StackPointerTracking, std::pair<int, int>>;

public:
  StackPointerTracking(BinaryFunction &BF,
                       MCPlusBuilder::AllocatorIdTy AllocatorId = 0);
  virtual ~StackPointerTracking() {}

  void run() { StackPointerTrackingBase<StackPointerTracking>::run(); }
};

} // end namespace bolt

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const std::pair<int, int> &Val);

} // end namespace llvm

#endif
