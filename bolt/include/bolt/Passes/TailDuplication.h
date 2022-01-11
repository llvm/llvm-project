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
  uint64_t PossibleDuplications = 0;

  /// Record how many times these duplications would get used.
  uint64_t PossibleDuplicationsDynamicCount = 0;

  /// Record the execution count of all unconditional branches.
  uint64_t UnconditionalBranchDynamicCount = 0;

  /// Record the execution count of all blocks.
  uint64_t AllBlocksDynamicCount = 0;

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

  /// Returns true if Reg is overwritten before its used by StartBB's sucessors
  bool isOverwrittenBeforeUsed(BinaryBasicBlock &StartBB, unsigned Reg) const;

  /// Constant and Copy Propagate for the block formed by OriginalBB and
  /// BlocksToPropagate
  void
  constantAndCopyPropagate(BinaryBasicBlock &OriginalBB,
                           std::vector<BinaryBasicBlock *> &BlocksToPropagate);

  /// True if Succ is in the same cache line as BB (approximately)
  bool isInCacheLine(const BinaryBasicBlock &BB,
                     const BinaryBasicBlock &Succ) const;

  /// Duplicates BlocksToDuplicate and places them after BB.
  std::vector<BinaryBasicBlock *>
  tailDuplicate(BinaryBasicBlock &BB,
                const std::vector<BinaryBasicBlock *> &BlocksToDuplicate) const;

  /// Returns a vector of BinaryBasicBlock to copy after BB. If it's empty,
  /// nothing should be duplicated
  std::vector<BinaryBasicBlock *>
  moderateCodeToDuplicate(BinaryBasicBlock &BB) const;
  std::vector<BinaryBasicBlock *>
  aggressiveCodeToDuplicate(BinaryBasicBlock &BB) const;

  void runOnFunction(BinaryFunction &Function);

public:
  explicit TailDuplication() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "tail duplication"; }

  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
