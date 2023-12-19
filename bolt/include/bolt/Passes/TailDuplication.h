//===- bolt/Passes/TailDuplication.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass founds cases when BBs have layout:
// #BB0:
// <body>
// jmp #BB2
// ....
// #BB1
// <body>
// #BB2:
// <body>
//
// And duplicates #BB2 and puts it after #BB0:
// #BB0:
// <body>
// #BB2:
// <body>
// ....
// #BB1
// <body>
// #BB2:
// <body>
//
// The advantage is getting rid of an unconditional branch and hopefully to
// improve i-cache performance by reducing fragmentation The disadvantage is
// that if there is too much code duplication, we may end up evicting hot cache
// lines and causing the opposite effect, hurting i-cache performance This needs
// to be well balanced to achieve the optimal effect
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_TAILDUPLICATION_H
#define BOLT_PASSES_TAILDUPLICATION_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

/// Pass for duplicating blocks that would require a jump.
class TailDuplication : public BinaryFunctionPass {
  /// Record how many possible tail duplications there can be.
  uint64_t ModifiedFunctions = 0;

  /// The number of duplicated basic blocks.
  uint64_t DuplicatedBlockCount = 0;

  /// The size (in bytes) of duplicated basic blocks.
  uint64_t DuplicatedByteCount = 0;

  /// Record how many times these duplications would get used.
  uint64_t DuplicationsDynamicCount = 0;

  /// Record the execution count of all blocks.
  uint64_t AllDynamicCount = 0;

  /// Record the number of instructions deleted because of propagation
  uint64_t StaticInstructionDeletionCount = 0;

  /// Record the number of instructions deleted because of propagation
  uint64_t DynamicInstructionDeletionCount = 0;

  /// Sets Regs with the caller saved registers
  void getCallerSavedRegs(const MCInst &Inst, BitVector &Regs,
                          BinaryContext &BC) const;

  /// Returns true if Reg is possibly overwritten by Inst
  bool regIsPossiblyOverwritten(const MCInst &Inst, unsigned Reg,
                                BinaryContext &BC) const;

  /// Returns true if Reg is definitely overwritten by Inst
  bool regIsDefinitelyOverwritten(const MCInst &Inst, unsigned Reg,
                                  BinaryContext &BC) const;

  /// Returns true if Reg is used by Inst
  bool regIsUsed(const MCInst &Inst, unsigned Reg, BinaryContext &BC) const;

  /// Returns true if Reg is overwritten before its used by StartBB's successors
  bool isOverwrittenBeforeUsed(BinaryBasicBlock &StartBB, unsigned Reg) const;

  /// Constant and Copy Propagate for the block formed by OriginalBB and
  /// BlocksToPropagate
  void
  constantAndCopyPropagate(BinaryBasicBlock &OriginalBB,
                           std::vector<BinaryBasicBlock *> &BlocksToPropagate);

  /// True if Tail is in the same cache line as BB (approximately)
  bool isInCacheLine(const BinaryBasicBlock &BB,
                     const BinaryBasicBlock &Tail) const;

  /// Duplicates BlocksToDuplicate and places them after BB.
  std::vector<BinaryBasicBlock *> duplicateBlocks(
      BinaryBasicBlock &BB,
      const std::vector<BinaryBasicBlock *> &BlocksToDuplicate) const;

  /// Decide whether the tail basic blocks should be duplicated after BB.
  bool shouldDuplicate(BinaryBasicBlock *BB, BinaryBasicBlock *Tail) const;

  /// Compute the cache score for a jump (Src, Dst) with frequency Count.
  /// The value is in the range [0..1] and quantifies how "cache-friendly"
  /// the jump is. The score is close to 1 for "short" forward jumps and
  /// it is 0 for "long" jumps exceeding a specified threshold; between the
  /// bounds, the value decreases linearly. For backward jumps, the value is
  /// scaled by a specified factor.
  double cacheScore(uint64_t SrcAddr, uint64_t SrcSize, uint64_t DstAddr,
                    uint64_t DstSize, uint64_t Count) const;

  /// Decide whether the cache score has been improved after duplication.
  bool cacheScoreImproved(const MCCodeEmitter *Emitter, BinaryFunction &BF,
                          BinaryBasicBlock *Pred, BinaryBasicBlock *Tail) const;

  /// A moderate strategy for tail duplication.
  /// Returns a vector of BinaryBasicBlock to copy after BB. If it's empty,
  /// nothing should be duplicated.
  std::vector<BinaryBasicBlock *>
  moderateDuplicate(BinaryBasicBlock &BB, BinaryBasicBlock &Tail) const;

  /// An aggressive strategy for tail duplication.
  std::vector<BinaryBasicBlock *>
  aggressiveDuplicate(BinaryBasicBlock &BB, BinaryBasicBlock &Tail) const;

  /// A cache-aware strategy for tail duplication.
  std::vector<BinaryBasicBlock *> cacheDuplicate(const MCCodeEmitter *Emitter,
                                                 BinaryFunction &BF,
                                                 BinaryBasicBlock *BB,
                                                 BinaryBasicBlock *Tail) const;

  void runOnFunction(BinaryFunction &Function);

public:
  enum DuplicationMode : char {
    TD_NONE = 0,
    TD_AGGRESSIVE,
    TD_MODERATE,
    TD_CACHE
  };

  explicit TailDuplication() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "tail duplication"; }

  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
