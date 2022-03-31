//===- bolt/Passes/JTFootprintReduction.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Jump table footprint reduction pass
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_JT_FOOTPRINT_REDUCTION_H
#define BOLT_PASSES_JT_FOOTPRINT_REDUCTION_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {
class DataflowInfoManager;

/// This pass identify indirect jumps to jump tables and reduce their entries
/// size from 8 to 4 bytes. For PIC jump tables, it will remove the PIC code
/// (since BOLT only process static code and it makes no sense to use expensive
/// PIC-style jumps in static code).
class JTFootprintReduction : public BinaryFunctionPass {
  uint64_t TotalJTScore{0};
  uint64_t TotalJTs{0};
  uint64_t TotalJTsDenied{0};
  uint64_t OptimizedScore{0};
  uint64_t IndJmps{0};
  uint64_t IndJmpsDenied{0};
  uint64_t NumJTsBadMatch{0};
  uint64_t NumJTsNoReg{0};
  uint64_t BytesSaved{0};
  DenseSet<JumpTable *> BlacklistedJTs;
  DenseSet<const BinaryFunction *> Modified;

  /// Check if \p Function presents jump tables where all jump locations can
  /// be safely changed to use a different code sequence. If this is true, we
  /// will be able to emit the whole table with a smaller entry size.
  void checkOpportunities(BinaryFunction &Function, DataflowInfoManager &Info);

  /// The Non-PIC jump table optimization consists of reducing the jump table
  /// entry size from 8 to 4 bytes. For that, we need to change the jump code
  /// sequence from a single jmp * instruction to a pair of load32zext-jmp
  /// instructions that depend on the availability of an extra register.
  /// This saves dcache/dTLB at the expense of icache.
  bool tryOptimizeNonPIC(BinaryContext &BC, BinaryBasicBlock &BB,
                         BinaryBasicBlock::iterator Inst, uint64_t JTAddr,
                         JumpTable *JumpTable, DataflowInfoManager &Info);

  /// The PIC jump table optimization consists of "de-pic-ifying" it, since the
  /// PIC jump sequence is larger than its non-PIC counterpart, saving icache.
  bool tryOptimizePIC(BinaryContext &BC, BinaryBasicBlock &BB,
                      BinaryBasicBlock::iterator Inst, uint64_t JTAddr,
                      JumpTable *JumpTable, DataflowInfoManager &Info);

  /// Run a pass for \p Function
  void optimizeFunction(BinaryFunction &Function, DataflowInfoManager &Info);

public:
  explicit JTFootprintReduction(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  /// BinaryPass interface functions
  const char *getName() const override { return "jt-footprint-reduction"; }
  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && Modified.count(&BF) > 0;
  }
  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
