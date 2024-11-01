//===- bolt/Passes/Instrumentation.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is an instrumentation pass that modifies the input binary to generate
// a profile after execution finishes. It can modify branches and calls to
// increment counters stored in the process memory. A runtime library is linked
// into the final binary to handle writing these counters to an fdata file. See
// runtime/instr.cpp
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_INSTRUMENTATION_H
#define BOLT_PASSES_INSTRUMENTATION_H

#include "bolt/Passes/BinaryPasses.h"
#include "bolt/Passes/InstrumentationSummary.h"
#include "llvm/Support/RWMutex.h"

namespace llvm {
namespace bolt {

class Instrumentation : public BinaryFunctionPass {
public:
  Instrumentation(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass),
        Summary(std::make_unique<InstrumentationSummary>()) {}

  /// Modifies all functions by inserting instrumentation code (first step)
  void runOnFunctions(BinaryContext &BC) override;

  const char *getName() const override { return "instrumentation"; }

private:
  void instrumentFunction(BinaryFunction &Function,
                          MCPlusBuilder::AllocatorIdTy = 0);

  /// Retrieve the string table index for the name of \p Function. We encode
  /// instrumented locations descriptions with the aid of a string table to
  /// manage memory of the instrumentation runtime in a more efficient way.
  /// If this function name is not represented in the string table yet, it will
  /// be inserted and its index returned.
  uint32_t getFunctionNameIndex(const BinaryFunction &Function);

  /// Metadata creation methods
  void createIndCallDescription(const BinaryFunction &FromFunction,
                                uint32_t From);
  void createIndCallTargetDescription(const BinaryFunction &ToFunction,
                                      uint32_t To);
  bool createCallDescription(FunctionDescription &FuncDesc,
                             const BinaryFunction &FromFunction, uint32_t From,
                             uint32_t FromNodeID,
                             const BinaryFunction &ToFunction, uint32_t To,
                             bool IsInvoke);
  bool createEdgeDescription(FunctionDescription &FuncDesc,
                             const BinaryFunction &FromFunction, uint32_t From,
                             uint32_t FromNodeID,
                             const BinaryFunction &ToFunction, uint32_t To,
                             uint32_t ToNodeID, bool Instrumented);
  void createLeafNodeDescription(FunctionDescription &FuncDesc, uint32_t Node);

  /// Create the sequence of instructions to increment a counter
  InstructionListType createInstrumentationSnippet(BinaryContext &BC,
                                                   bool IsLeaf);

  // Critical edges worklist
  // This worklist keeps track of CFG edges <From-To> that needs to be split.
  // This task is deferred until we finish processing all BBs because we can't
  // modify the CFG while iterating over it. For each edge, \p SplitInstrsTy
  // stores the list of instrumentation instructions as a vector of MCInsts.
  // instrumentOneTarget() populates this, instrumentFunction() consumes.
  using SplitWorklistTy =
      std::vector<std::pair<BinaryBasicBlock *, BinaryBasicBlock *>>;
  using SplitInstrsTy = std::vector<InstructionListType>;

  /// Instrument the branch or call in \p Iter. \p TargetBB should be non-null
  /// if this is a local branch and null if it is a call. Return true if the
  /// location was instrumented with an explicit counter or false if it just
  /// created the description, but no explicit counters were necessary.
  bool instrumentOneTarget(SplitWorklistTy &SplitWorklist,
                           SplitInstrsTy &SplitInstrs,
                           BinaryBasicBlock::iterator &Iter,
                           BinaryFunction &FromFunction,
                           BinaryBasicBlock &FromBB, uint32_t From,
                           BinaryFunction &ToFunc, BinaryBasicBlock *TargetBB,
                           uint32_t ToOffset, bool IsLeaf, bool IsInvoke,
                           FunctionDescription *FuncDesc, uint32_t FromNodeID,
                           uint32_t ToNodeID = 0);

  void instrumentLeafNode(BinaryBasicBlock &BB, BinaryBasicBlock::iterator Iter,
                          bool IsLeaf, FunctionDescription &FuncDesc,
                          uint32_t Node);

  void instrumentIndirectTarget(BinaryBasicBlock &BB,
                                BinaryBasicBlock::iterator &Iter,
                                BinaryFunction &FromFunction, uint32_t From);

  void createAuxiliaryFunctions(BinaryContext &BC);

  uint32_t getFDSize() const;

  /// Create a runtime library, pass the BinData over, and register it
  /// under \p BC.
  void setupRuntimeLibrary(BinaryContext &BC);

  /// strtab indices in StringTable for each function name
  std::unordered_map<const BinaryFunction *, uint32_t> FuncToStringIdx;

  mutable llvm::sys::RWMutex FDMutex;

  /// The data generated during Instrumentation pass that needs to
  /// be passed to the Instrument runtime library.
  std::unique_ptr<InstrumentationSummary> Summary;

  /// Statistics on counters
  uint32_t DirectCallCounters{0};
  uint32_t BranchCounters{0};
  uint32_t LeafNodeCounters{0};

  /// Indirect call instrumentation functions
  BinaryFunction *IndCallHandlerExitBBFunction;
  BinaryFunction *IndTailCallHandlerExitBBFunction;
};
} // namespace bolt
} // namespace llvm

#endif
