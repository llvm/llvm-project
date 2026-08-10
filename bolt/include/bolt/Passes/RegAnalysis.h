//===- bolt/Passes/RegAnalysis.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_REGANALYSIS_H
#define BOLT_PASSES_REGANALYSIS_H

#include "llvm/ADT/BitVector.h"
#include <cstdint>
#include <map>

namespace llvm {
class MCInst;

namespace bolt {
class BinaryContext;
class BinaryFunction;
class BinaryFunctionCallGraph;

/// Determine the set of registers read or clobbered for each instruction
/// in a BinaryFunction. If the instruction is a call, this analysis rely on
/// a call graph traversal to accurately extract the set of registers touched
/// after the call returns.
class RegAnalysis {
public:
  /// Compute the set of registers \p Func may read from during its execution.
  BitVector getFunctionUsedRegsList(const BinaryFunction *Func);

  /// Compute the set of registers \p Func may write to during its execution,
  /// starting at the point when it is called up until when it returns. Returns
  /// a BitVector the size of the target number of registers, representing the
  /// set of clobbered registers.
  BitVector getFunctionClobberList(const BinaryFunction *Func);

  RegAnalysis(BinaryContext &BC, std::map<uint64_t, BinaryFunction> *BFs,
              BinaryFunctionCallGraph *CG);

  /// Compute the set of registers \p Inst may read from, marking them in
  /// \p RegSet. If GetClobbers is true, the set set the instr may write to.
  /// Use the callgraph to fill out this info for calls.
  void getInstUsedRegsList(const MCInst &Inst, BitVector &RegSet,
                           bool GetClobbers) const;

  /// Compute the set of registers \p Inst may write to, marking them in
  /// \p KillSet. If this is a call, try to get the set of registers the call
  /// target will write to.
  void getInstClobberList(const MCInst &Inst, BitVector &KillSet) const;

  /// Return true iff Vec has a conservative estimation of used/clobbered regs,
  /// expressing no specific knowledge of reg usage.
  bool isConservative(BitVector &Vec) const;

  /// Set what to do when lacking information about a call
  enum class ConservativeStrategy { CLOBBERS_ALL, CLOBBERS_ABI, CLOBBERS_NONE };
  void setConservativeStrategy(ConservativeStrategy S) { CS = S; }

  /// Print stats about the quality of our analysis
  void printStats();

private:
  BinaryContext &BC;

  /// Map functions to the set of registers they may overwrite starting at when
  /// it is called until it returns to the caller.
  std::map<const BinaryFunction *, BitVector> RegsKilledMap;

  /// Similar concept above but for registers that are read in that function.
  std::map<const BinaryFunction *, BitVector> RegsGenMap;

  /// Analysis stats counters
  uint64_t NumFunctionsAllClobber{0};
  uint64_t CountFunctionsAllClobber{0};
  uint64_t CountDenominator{0};

  ConservativeStrategy CS;

  /// Helper function used to get the set of clobbered/used regs whenever
  /// we know nothing about the function.
  void beConservative(BitVector &Result) const;
};

} // namespace bolt
} // namespace llvm

#endif
