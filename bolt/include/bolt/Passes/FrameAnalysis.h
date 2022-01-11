//===- bolt/Passes/FrameAnalysis.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_FRAMEANALYSIS_H
#define BOLT_PASSES_FRAMEANALYSIS_H

#include "bolt/Passes/StackPointerTracking.h"

namespace llvm {
namespace bolt {
class BinaryFunctionCallGraph;

/// Alias analysis information attached to each instruction that accesses a
/// frame position. This is called a "frame index" by LLVM Target libs when
/// it is building a MachineFunction frame, and we use the same name here
/// because we are essentially doing the job of frame reconstruction.
struct FrameIndexEntry {
  /// If both IsLoad and IsStore are set, it means this is an instruction that
  /// reads and updates this frame location.
  bool IsLoad;
  bool IsStore;
  /// If a store, this controls whether the store uses a register os an imm
  /// as the source value.
  bool IsStoreFromReg;
  /// If load, this holds the destination register. If store, this holds
  /// either the source register or source immediate.
  int32_t RegOrImm;

  /// StackOffset and Size are the two aspects that identify this frame access
  /// for the purposes of alias analysis.
  int64_t StackOffset;
  uint8_t Size;

  /// If this is false, we will never atempt to remove or optimize this
  /// instruction. We just use it to keep track of stores we don't fully
  /// understand but we know it may write to a frame position.
  bool IsSimple;

  uint16_t StackPtrReg;
};

/// Record an access to an argument in stack. This should be attached to
/// call instructions, so StackOffset and Size are determined in the context
/// of the caller. This information helps the caller understand how the callee
/// may access its private stack.
struct ArgInStackAccess {
  int64_t StackOffset;
  uint8_t Size;

  bool operator<(const ArgInStackAccess &RHS) const {
    if (StackOffset != RHS.StackOffset)
      return StackOffset < RHS.StackOffset;
    return Size < RHS.Size;
  }
};

/// The set of all args-in-stack accesses for a given instruction. If
/// AssumeEverything is true, then the set should be ignored and the
/// corresponding instruction should be treated as accessing the entire
/// stack for the purposes of analysis and optimization.
struct ArgAccesses {
  bool AssumeEverything;
  std::set<ArgInStackAccess> Set;

  explicit ArgAccesses(bool AssumeEverything)
      : AssumeEverything(AssumeEverything) {}
};

raw_ostream &operator<<(raw_ostream &OS, const FrameIndexEntry &FIE);

/// This pass attaches stack access information to instructions. If a load/store
/// instruction accesses a stack position, it will identify the CFA offset and
/// size information of this access, where CFA is the Canonical Frame Address
/// (using DWARF terminology).
///
/// This pass also computes frame usage information obtained by a bottom-up call
/// graph traversal: which registers are clobbered by functions (including their
/// callees as determined by the call graph), whether a function accesses its
/// caller's stack frame and whether a function demands its stack to be aligned
/// due to the use of SSE aligned load/store operations present in itself or any
/// of its direct or indirect callees.
///
/// Initialization:
///
///   FrameAnalysis FA(PrintPass);
///   FA.runOnFunctions(BC);
///
/// Usage (fetching frame access information about a given instruction):
///
///   auto FIE = FA.getFIEFor(BC, Instruction);
///   if (FIE && FIE->IsSimple) {
///     ... = FIE->StackOffset
///     ... = FIE->Size
///   }
///
/// Usage (determining the set of stack positions accessed by the target of a
/// call:
///
///    auto Args = FA.getArgAccessesFor(BC, CallInst);
///    if (Args && Args->AssumeEverything) {
///      ... callee may access any position of our current stack frame
///    }
///
class FrameAnalysis {
  BinaryContext &BC;

  /// Map functions to the set of <stack offsets, size> tuples representing
  /// accesses to stack positions that belongs to caller
  std::map<const BinaryFunction *, std::set<std::pair<int64_t, uint8_t>>>
      ArgsTouchedMap;

  /// The set of functions we were able to perform the full analysis up to
  /// restoring frame indexes for all load/store instructions.
  DenseSet<const BinaryFunction *> AnalyzedFunctions;

  /// Set of functions that require the stack to be 16B aligned
  DenseSet<const BinaryFunction *> FunctionsRequireAlignment;

  /// Owns ArgAccesses for all instructions. References to elements are
  /// attached to instructions as indexes to this vector, in MCAnnotations.
  std::vector<ArgAccesses> ArgAccessesVector;
  /// Same for FrameIndexEntries.
  std::vector<FrameIndexEntry> FIEVector;

  /// Analysis stats counters
  uint64_t NumFunctionsNotOptimized{0};
  uint64_t NumFunctionsFailedRestoreFI{0};
  uint64_t CountFunctionsNotOptimized{0};
  uint64_t CountFunctionsFailedRestoreFI{0};
  uint64_t CountDenominator{0};

  /// Convenience functions for appending MCAnnotations to instructions with
  /// our specific data
  void addArgAccessesFor(MCInst &Inst, ArgAccesses &&AA);
  void addArgInStackAccessFor(MCInst &Inst, const ArgInStackAccess &Arg);
  void addFIEFor(MCInst &Inst, const FrameIndexEntry &FIE);

  /// Perform the step of building the set of registers clobbered by each
  /// function execution, populating RegsKilledMap and RegsGenMap.
  void traverseCG(BinaryFunctionCallGraph &CG);

  /// Analyzes an instruction and if it is a call, checks the called function
  /// to record which args in stack are accessed, if any. Returns true if
  /// the args data associated with this instruction were updated.
  bool updateArgsTouchedFor(const BinaryFunction &BF, MCInst &Inst,
                            int CurOffset);

  /// Performs a pass over \p BF to check for accesses to arguments in stack,
  /// flagging those as accessing the caller stack frame. All functions called
  /// by \p BF must have been previously analyzed. Returns true if updated
  /// args data about this function.
  bool computeArgsAccessed(BinaryFunction &BF);

  /// Alias analysis to disambiguate which frame position is accessed by each
  /// instruction in function \p BF. Add MCAnnotation<FrameIndexEntry> to
  /// instructions that access a frame position. Return false if it failed
  /// to analyze and this information can't be safely determined for \p BF.
  bool restoreFrameIndex(BinaryFunction &BF);

  /// A store for SPT info per function
  std::unordered_map<const BinaryFunction *,
                     std::unique_ptr<StackPointerTracking>>
      SPTMap;

  /// A vector that stores ids of the allocators that are used in SPT
  /// computation
  std::vector<MCPlusBuilder::AllocatorIdTy> SPTAllocatorsId;

public:
  explicit FrameAnalysis(BinaryContext &BC, BinaryFunctionCallGraph &CG);

  /// Return true if we could fully analyze \p Func
  bool hasFrameInfo(const BinaryFunction &Func) const {
    return AnalyzedFunctions.count(&Func);
  }

  /// Return true if \p Func cannot operate with a misaligned CFA
  bool requiresAlignment(const BinaryFunction &Func) const {
    return FunctionsRequireAlignment.count(&Func);
  }

  /// Functions for retrieving our specific MCAnnotation data from instructions
  ErrorOr<ArgAccesses &> getArgAccessesFor(const MCInst &Inst);

  ErrorOr<const ArgAccesses &> getArgAccessesFor(const MCInst &Inst) const;

  ErrorOr<const FrameIndexEntry &> getFIEFor(const MCInst &Inst) const;

  /// Remove all MCAnnotations attached by this pass
  void cleanAnnotations();

  ~FrameAnalysis() { cleanAnnotations(); }

  /// Print to standard output statistics about the analysis performed by this
  /// pass
  void printStats();

  /// Get or create an SPT object and run the analysis
  StackPointerTracking &getSPT(BinaryFunction &BF) {
    if (!SPTMap.count(&BF)) {
      SPTMap.emplace(&BF, std::make_unique<StackPointerTracking>(BF));
      auto Iter = SPTMap.find(&BF);
      assert(Iter != SPTMap.end() && "item should exist");
      Iter->second->run();
      return *Iter->second;
    }

    auto Iter = SPTMap.find(&BF);
    assert(Iter != SPTMap.end() && "item should exist");
    return *Iter->second;
  }

  /// Clean and de-allocate all SPT objects
  void clearSPTMap();

  /// Perform SPT analysis for all functions in parallel
  void preComputeSPT();
};

} // namespace bolt
} // namespace llvm

#endif
