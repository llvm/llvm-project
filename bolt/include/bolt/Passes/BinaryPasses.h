//===- bolt/Passes/BinaryPasses.h - Binary-level passes ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The set of optimization/analysis passes that run on BinaryFunctions.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_BINARY_PASSES_H
#define BOLT_PASSES_BINARY_PASSES_H

#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/DynoStats.h"
#include "llvm/Support/CommandLine.h"
#include <atomic>
#include <map>
#include <set>
#include <string>
#include <unordered_set>

namespace llvm {
namespace bolt {

/// An optimization/analysis pass that runs on functions.
class BinaryFunctionPass {
protected:
  bool PrintPass;

  explicit BinaryFunctionPass(const bool PrintPass) : PrintPass(PrintPass) {}

  /// Control whether a specific function should be skipped during
  /// optimization.
  virtual bool shouldOptimize(const BinaryFunction &BF) const;

public:
  virtual ~BinaryFunctionPass() = default;

  /// The name of this pass
  virtual const char *getName() const = 0;

  /// Control whether debug info is printed after this pass is completed.
  bool printPass() const { return PrintPass; }

  /// Control whether debug info is printed for an individual function after
  /// this pass is completed (printPass() must have returned true).
  virtual bool shouldPrint(const BinaryFunction &BF) const;

  /// Execute this pass on the given functions.
  virtual void runOnFunctions(BinaryContext &BC) = 0;
};

/// A pass to print program-wide dynostats.
class DynoStatsPrintPass : public BinaryFunctionPass {
protected:
  DynoStats PrevDynoStats;
  std::string Title;

public:
  DynoStatsPrintPass(const DynoStats &PrevDynoStats, const char *Title)
      : BinaryFunctionPass(false), PrevDynoStats(PrevDynoStats), Title(Title) {}

  const char *getName() const override {
    return "print dyno-stats after optimizations";
  }

  bool shouldPrint(const BinaryFunction &BF) const override { return false; }

  void runOnFunctions(BinaryContext &BC) override {
    const DynoStats NewDynoStats = getDynoStats(BC.getBinaryFunctions());
    const bool Changed = (NewDynoStats != PrevDynoStats);
    outs() << "BOLT-INFO: program-wide dynostats " << Title
           << (Changed ? "" : " (no change)") << ":\n\n"
           << PrevDynoStats;
    if (Changed) {
      outs() << '\n';
      NewDynoStats.print(outs(), &PrevDynoStats, BC.InstPrinter.get());
    }
    outs() << '\n';
  }
};

/// The pass normalizes CFG by performing the following transformations:
///   * removes empty basic blocks
///   * merges duplicate edges and updates jump instructions
class NormalizeCFG : public BinaryFunctionPass {
  std::atomic<uint64_t> NumBlocksRemoved{0};
  std::atomic<uint64_t> NumDuplicateEdgesMerged{0};

  void runOnFunction(BinaryFunction &BF);

public:
  NormalizeCFG(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "normalize CFG"; }

  void runOnFunctions(BinaryContext &) override;
};

/// Detect and eliminate unreachable basic blocks. We could have those
/// filled with nops and they are used for alignment.
class EliminateUnreachableBlocks : public BinaryFunctionPass {
  std::unordered_set<const BinaryFunction *> Modified;
  std::atomic<unsigned> DeletedBlocks{0};
  std::atomic<uint64_t> DeletedBytes{0};
  void runOnFunction(BinaryFunction &Function);

public:
  EliminateUnreachableBlocks(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "eliminate-unreachable"; }
  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && Modified.count(&BF) > 0;
  }
  void runOnFunctions(BinaryContext &) override;
};

// Reorder the basic blocks for each function based on hotness.
class ReorderBasicBlocks : public BinaryFunctionPass {
public:
  /// Choose which strategy should the block layout heuristic prioritize when
  /// facing conflicting goals.
  enum LayoutType : char {
    /// LT_NONE - do not change layout of basic blocks
    LT_NONE = 0, /// no reordering
    /// LT_REVERSE - reverse the order of basic blocks, meant for testing
    /// purposes. The first basic block is left intact and the rest are
    /// put in the reverse order.
    LT_REVERSE,
    /// LT_OPTIMIZE - optimize layout of basic blocks based on profile.
    LT_OPTIMIZE,
    /// LT_OPTIMIZE_BRANCH is an implementation of what is suggested in Pettis'
    /// paper (PLDI '90) about block reordering, trying to minimize branch
    /// mispredictions.
    LT_OPTIMIZE_BRANCH,
    /// LT_OPTIMIZE_CACHE piggybacks on the idea from Ispike paper (CGO '04)
    /// that suggests putting frequently executed chains first in the layout.
    LT_OPTIMIZE_CACHE,
    /// Block reordering guided by the extended TSP metric.
    LT_OPTIMIZE_EXT_TSP,
    /// Create clusters and use random order for them.
    LT_OPTIMIZE_SHUFFLE,
  };

private:
  void modifyFunctionLayout(BinaryFunction &Function, LayoutType Type,
                            bool MinBranchClusters) const;

public:
  explicit ReorderBasicBlocks(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  bool shouldOptimize(const BinaryFunction &BF) const override;

  const char *getName() const override { return "reorder-blocks"; }
  bool shouldPrint(const BinaryFunction &BF) const override;
  void runOnFunctions(BinaryContext &BC) override;
};

/// Sync local branches with CFG.
class FixupBranches : public BinaryFunctionPass {
public:
  explicit FixupBranches(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "fix-branches"; }
  void runOnFunctions(BinaryContext &BC) override;
};

/// Fix the CFI state and exception handling information after all other
/// passes have completed.
class FinalizeFunctions : public BinaryFunctionPass {
public:
  explicit FinalizeFunctions(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "finalize-functions"; }
  void runOnFunctions(BinaryContext &BC) override;
};

/// Perform any necessary adjustments for functions that do not fit into their
/// original space in non-relocation mode.
class CheckLargeFunctions : public BinaryFunctionPass {
public:
  explicit CheckLargeFunctions(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "check-large-functions"; }

  void runOnFunctions(BinaryContext &BC) override;

  bool shouldOptimize(const BinaryFunction &BF) const override;
};

/// Convert and remove all BOLT-related annotations before LLVM code emission.
class LowerAnnotations : public BinaryFunctionPass {
public:
  explicit LowerAnnotations(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "lower-annotations"; }
  void runOnFunctions(BinaryContext &BC) override;
};

/// An optimization to simplify conditional tail calls by removing
/// unnecessary branches.
///
/// This optimization considers both of the following cases:
///
/// foo: ...
///      jcc L1   original
///      ...
/// L1:  jmp bar  # TAILJMP
///
/// ->
///
/// foo: ...
///      jcc bar  iff jcc L1 is expected
///      ...
///
/// L1 is unreachable
///
/// OR
///
/// foo: ...
///      jcc  L2
/// L1:  jmp  dest  # TAILJMP
/// L2:  ...
///
/// ->
///
/// foo: jncc dest  # TAILJMP
/// L2:  ...
///
/// L1 is unreachable
///
/// For this particular case, the first basic block ends with
/// a conditional branch and has two successors, one fall-through
/// and one for when the condition is true.
/// The target of the conditional is a basic block with a single
/// unconditional branch (i.e. tail call) to another function.
/// We don't care about the contents of the fall-through block.
/// We assume that the target of the conditional branch is the
/// first successor.
class SimplifyConditionalTailCalls : public BinaryFunctionPass {
  uint64_t NumCandidateTailCalls{0};
  uint64_t NumTailCallsPatched{0};
  uint64_t CTCExecCount{0};
  uint64_t CTCTakenCount{0};
  uint64_t NumOrigForwardBranches{0};
  uint64_t NumOrigBackwardBranches{0};
  uint64_t NumDoubleJumps{0};
  uint64_t DeletedBlocks{0};
  uint64_t DeletedBytes{0};
  std::unordered_set<const BinaryFunction *> Modified;
  std::set<const BinaryBasicBlock *> BeenOptimized;

  bool shouldRewriteBranch(const BinaryBasicBlock *PredBB,
                           const MCInst &CondBranch, const BinaryBasicBlock *BB,
                           const bool DirectionFlag);

  uint64_t fixTailCalls(BinaryFunction &BF);

public:
  explicit SimplifyConditionalTailCalls(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override {
    return "simplify-conditional-tail-calls";
  }
  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && Modified.count(&BF) > 0;
  }
  void runOnFunctions(BinaryContext &BC) override;
};

/// Convert instructions to the form with the minimum operand width.
class ShortenInstructions : public BinaryFunctionPass {
  uint64_t shortenInstructions(BinaryFunction &Function);

public:
  explicit ShortenInstructions(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "shorten-instructions"; }

  void runOnFunctions(BinaryContext &BC) override;
};

/// Perform simple peephole optimizations.
class Peepholes : public BinaryFunctionPass {
  uint64_t NumDoubleJumps{0};
  uint64_t TailCallTraps{0};
  uint64_t NumUselessCondBranches{0};

  /// Add trap instructions immediately after indirect tail calls to prevent
  /// the processor from decoding instructions immediate following the
  /// tailcall.
  void addTailcallTraps(BinaryFunction &Function);

  /// Remove useless duplicate successors.  When the conditional
  /// successor is the same as the unconditional successor, we can
  /// remove the conditional successor and branch instruction.
  void removeUselessCondBranches(BinaryFunction &Function);

public:
  explicit Peepholes(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "peepholes"; }
  void runOnFunctions(BinaryContext &BC) override;
};

/// An optimization to simplify loads from read-only sections.The pass converts
/// load instructions with statically computed target address such as:
///
///      mov 0x12f(%rip), %eax
///
/// to their counterparts that use immediate operands instead of memory loads:
///
///     mov $0x4007dc, %eax
///
/// when the target address points somewhere inside a read-only section.
///
class SimplifyRODataLoads : public BinaryFunctionPass {
  uint64_t NumLoadsSimplified{0};
  uint64_t NumDynamicLoadsSimplified{0};
  uint64_t NumLoadsFound{0};
  uint64_t NumDynamicLoadsFound{0};
  std::unordered_set<const BinaryFunction *> Modified;

  bool simplifyRODataLoads(BinaryFunction &BF);

public:
  explicit SimplifyRODataLoads(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "simplify-read-only-loads"; }
  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && Modified.count(&BF) > 0;
  }
  void runOnFunctions(BinaryContext &BC) override;
};

/// Assign output sections to all functions.
class AssignSections : public BinaryFunctionPass {
public:
  explicit AssignSections() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "assign-sections"; }
  void runOnFunctions(BinaryContext &BC) override;
};

/// Compute and report to the user the imbalance in flow equations for all
/// CFGs, so we can detect bad quality profile. Prints average and standard
/// deviation of the absolute differences of outgoing flow minus incoming flow
/// for blocks of interest (excluding prologues, epilogues, and BB frequency
/// lower than 100).
class PrintProfileStats : public BinaryFunctionPass {
public:
  explicit PrintProfileStats(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "profile-stats"; }
  bool shouldPrint(const BinaryFunction &) const override { return false; }
  void runOnFunctions(BinaryContext &BC) override;
};

/// Prints a list of the top 100 functions sorted by a set of
/// dyno stats categories.
class PrintProgramStats : public BinaryFunctionPass {
public:
  explicit PrintProgramStats(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "print-stats"; }
  bool shouldPrint(const BinaryFunction &) const override { return false; }
  void runOnFunctions(BinaryContext &BC) override;
};

/// Pass for lowering any instructions that we have raised and that have
/// to be lowered.
class InstructionLowering : public BinaryFunctionPass {
public:
  explicit InstructionLowering(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "inst-lowering"; }

  void runOnFunctions(BinaryContext &BC) override;
};

/// Pass for stripping 'repz' from 'repz retq' sequence of instructions.
class StripRepRet : public BinaryFunctionPass {
public:
  explicit StripRepRet(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "strip-rep-ret"; }

  void runOnFunctions(BinaryContext &BC) override;
};

/// Pass for inlining calls to memcpy using 'rep movsb' on X86.
class InlineMemcpy : public BinaryFunctionPass {
public:
  explicit InlineMemcpy(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "inline-memcpy"; }

  void runOnFunctions(BinaryContext &BC) override;
};

/// Pass for specializing memcpy for a size of 1 byte.
class SpecializeMemcpy1 : public BinaryFunctionPass {
private:
  std::vector<std::string> Spec;

  /// Return indices of the call sites to optimize. Count starts at 1.
  /// Returns an empty set for all call sites in the function.
  std::set<size_t> getCallSitesToOptimize(const BinaryFunction &) const;

public:
  explicit SpecializeMemcpy1(const cl::opt<bool> &PrintPass,
                             cl::list<std::string> &Spec)
      : BinaryFunctionPass(PrintPass), Spec(Spec) {}

  bool shouldOptimize(const BinaryFunction &BF) const override;

  const char *getName() const override { return "specialize-memcpy"; }

  void runOnFunctions(BinaryContext &BC) override;
};

/// Pass to remove nops in code
class RemoveNops : public BinaryFunctionPass {
  void runOnFunction(BinaryFunction &Function);

public:
  explicit RemoveNops(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "remove-nops"; }

  /// Pass entry point
  void runOnFunctions(BinaryContext &BC) override;
};

enum FrameOptimizationType : char {
  FOP_NONE, /// Don't perform FOP.
  FOP_HOT,  /// Perform FOP on hot functions.
  FOP_ALL   /// Perform FOP on all functions.
};

} // namespace bolt
} // namespace llvm

#endif
