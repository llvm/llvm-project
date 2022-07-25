//===- bolt/Passes/ValidateInternalCalls.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_VALIDATEINTERNALCALLS_H
#define BOLT_PASSES_VALIDATEINTERNALCALLS_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

/// Post-processing for internal calls. What are those? They are call
/// instructions that do not transfer control to another function, but
/// rather branch to a basic block inside the caller function itself.
/// This pass checks that the internal calls observed in a function are
/// manageable. We support two types:
///
///   1. Position Independent Code (PIC) tricks: in this type of internal
///      call, we don't really have a call because the return address is
///      not utilized for branching to, but only as a base address to
///      reference other objects. We call it a "trick" because this is not
///      the standard way a compiler would do this and this will often come
///      from awkwardly written assembly code.
///
///   2. Real internal calls: in this case, a function was inlined inside
///      a caller, but the CALL instruction wasn't removed. This pair of
///      caller-callee is treated as a single function and is analyzed
///      here.
///
/// In general, the rest of the BOLT pipeline (other optimizations, including
/// code reordering) will not support neither of these cases. In this pass,
/// we just identify them, verify they are safe (do not reference objects
/// that will be moved after reordering) and freeze these functions in the
/// way they were read. We do this by marking them as non-simple.
///
/// Why do we freeze them?
///
/// Type 1 is not safe to optimize because any changed offsets will break the
/// PIC references made in this code. Type 2 is not safe to optimize because
/// it requires BOLT to understand a new CFG format where internal calls are
/// broken into two BBs (calling block and returning block), and we currently do
/// not support this  elsewhere. Only this pass is able to make sense of these
/// non-canonical CFGs (specifically, fixBranches does not support them).
///
class ValidateInternalCalls : public BinaryFunctionPass {
public:
  explicit ValidateInternalCalls(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "validate-internal-calls"; }

  void runOnFunctions(BinaryContext &BC) override;

private:
  /// Fix the CFG to take into consideration internal calls that do not
  /// return, but are only used as a trick to perform Position Independent
  /// Code (PIC) computations. This will change internal calls to be treated
  /// as unconditional jumps.
  void fixCFGForPIC(BinaryFunction &Function) const;

  /// Fix the CFG to take into consideration real internal calls (whole
  /// functions that got inlined inside its caller, but the CALL instruction
  /// wasn't removed).
  bool fixCFGForIC(BinaryFunction &Function) const;

  /// Detect tail calls in the range of the PIC access and fail to validate if
  /// one is detected. Tail calls are dangerous because they may be emitted
  /// with a different size in comparison with the original code.
  /// FIXME: shortenInstructions and NOP sizes can impact offsets too
  bool hasTailCallsInRange(BinaryFunction &Function) const;

  /// Check that the PIC computations performed by Type 1 internal calls are
  /// safe
  bool analyzeFunction(BinaryFunction &Function) const;

  /// The annotation tag we use to keep track of internal calls we already
  /// processed.
  StringRef getProcessedICTag() const { return "ProcessedInternalCall"; }

  void clearAnnotations(BinaryFunction &Function) const {
    const BinaryContext &BC = Function.getBinaryContext();
    for (BinaryBasicBlock &BB : Function)
      for (MCInst &Inst : BB)
        BC.MIB->removeAnnotation(Inst, getProcessedICTag());
  }
};

} // namespace bolt
} // namespace llvm

#endif
