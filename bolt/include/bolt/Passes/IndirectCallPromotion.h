//===- bolt/Passes/IndirectCallPromotion.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The indirect call promotion (ICP) optimization pass.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_INDIRECT_CALL_PROMOTION_H
#define BOLT_PASSES_INDIRECT_CALL_PROMOTION_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

/// Optimize indirect calls.
/// The indirect call promotion pass visits each indirect call and
/// examines a branch profile for each. If the most frequent targets
/// from that callsite exceed the specified threshold (default 90%),
/// the call is promoted. Otherwise, it is ignored. By default,
/// only one target is considered at each callsite.
///
/// When an candidate callsite is processed, we modify the callsite
/// to test for the most common call targets before calling through
/// the original generic call mechanism.
///
/// The CFG and layout are modified by ICP.
///
/// A few new command line options have been added:
///   -indirect-call-promotion=[none,call,jump-tables,all]
///   -indirect-call-promotion-threshold=<percentage>
///   -indirect-call-promotion-mispredict-threshold=<percentage>
///   -indirect-call-promotion-topn=<int>
///
/// The threshold is the minimum frequency of a call target needed
/// before ICP is triggered.
///
/// The mispredict threshold is used to disable the optimization at
/// any callsite where the branch predictor does a good enough job
/// that ICP wouldn't help regardless of the frequency of the most
/// common target.
///
/// The topn option controls the number of targets to consider for
/// each callsite, e.g. ICP is triggered if topn=2 and the total
/// frequency of the top two call targets exceeds the threshold.
///
/// The minimize code size option controls whether or not the hot
/// calls are to registers (callq %r10) or to function addresses
/// (callq $foo).
///
/// Example of ICP:
///
/// C++ code:
///
///   int B_count = 0;
///   int C_count = 0;
///
///   struct A { virtual void foo() = 0; }
///   struct B : public A { virtual void foo() { ++B_count; }; };
///   struct C : public A { virtual void foo() { ++C_count; }; };
///
///   A* a = ...
///   a->foo();
///   ...
///
/// original assembly:
///
///   B0: 49 8b 07             mov    (%r15),%rax
///       4c 89 ff             mov    %r15,%rdi
///       ff 10                callq  *(%rax)
///       41 83 e6 01          and    $0x1,%r14d
///       4d 89 e6             mov    %r12,%r14
///       4c 0f 44 f5          cmove  %rbp,%r14
///       4c 89 f7             mov    %r14,%rdi
///       ...
///
/// after ICP:
///
///   B0: 49 8b 07             mov    (%r15),%rax
///       4c 89 ff             mov    %r15,%rdi
///       48 81 38 e0 0b 40 00 cmpq   $B::foo,(%rax)
///       75 29                jne    B3
///   B1: e8 45 03 00 00       callq  $B::foo
///   B2: 41 83 e6 01          and    $0x1,%r14d
///       4d 89 e6             mov    %r12,%r14
///       4c 0f 44 f5          cmove  %rbp,%r14
///       4c 89 f7             mov    %r14,%rdi
///       ...
///
///   B3: ff 10                callq  *(%rax)
///       eb d6                jmp    B2
///
class IndirectCallPromotion : public BinaryFunctionPass {
  using BasicBlocksVector = std::vector<std::unique_ptr<BinaryBasicBlock>>;
  using MethodInfoType = std::pair<std::vector<std::pair<MCSymbol *, uint64_t>>,
                                   std::vector<MCInst *>>;
  using JumpTableInfoType = std::vector<std::pair<uint64_t, uint64_t>>;
  using SymTargetsType = std::vector<std::pair<MCSymbol *, uint64_t>>;
  struct Location {
    MCSymbol *Sym{nullptr};
    uint64_t Addr{0};
    bool isValid() const { return Sym || (!Sym && Addr != 0); }
    Location() {}
    explicit Location(MCSymbol *Sym) : Sym(Sym) {}
    explicit Location(uint64_t Addr) : Addr(Addr) {}
  };

  struct Callsite {
    Location From;
    Location To;
    uint64_t Mispreds{0};
    uint64_t Branches{0};
    // Indices in the jmp table (jt only)
    std::vector<uint64_t> JTIndices;
    bool isValid() const { return From.isValid() && To.isValid(); }
    Callsite(BinaryFunction &BF, const IndirectCallProfile &ICP);
    Callsite(const Location &From, const Location &To, uint64_t Mispreds,
             uint64_t Branches, uint64_t JTIndex)
        : From(From), To(To), Mispreds(Mispreds), Branches(Branches),
          JTIndices(1, JTIndex) {}
  };

  std::unordered_set<const BinaryFunction *> Modified;
  // Total number of calls from all callsites.
  uint64_t TotalCalls{0};

  // Total number of indirect calls from all callsites.
  // (a fraction of TotalCalls)
  uint64_t TotalIndirectCalls{0};

  // Total number of jmp table calls from all callsites.
  // (a fraction of TotalCalls)
  uint64_t TotalIndirectJmps{0};

  // Total number of callsites that use indirect calls.
  // (the total number of callsites is not recorded)
  uint64_t TotalIndirectCallsites{0};

  // Total number of callsites that are jump tables.
  uint64_t TotalJumpTableCallsites{0};

  // Total number of indirect callsites that are optimized by ICP.
  // (a fraction of TotalIndirectCallsites)
  uint64_t TotalOptimizedIndirectCallsites{0};

  // Total number of method callsites that can have loads eliminated.
  mutable uint64_t TotalMethodLoadEliminationCandidates{0};

  // Total number of method callsites that had loads eliminated.
  uint64_t TotalMethodLoadsEliminated{0};

  // Total number of jump table callsites that are optimized by ICP.
  uint64_t TotalOptimizedJumpTableCallsites{0};

  // Total number of indirect calls that are optimized by ICP.
  // (a fraction of TotalCalls)
  uint64_t TotalNumFrequentCalls{0};

  // Total number of jump table calls that are optimized by ICP.
  // (a fraction of TotalCalls)
  uint64_t TotalNumFrequentJmps{0};

  // Total number of jump table sites that can use hot indices.
  mutable uint64_t TotalIndexBasedCandidates{0};

  // Total number of jump table sites that use hot indices.
  uint64_t TotalIndexBasedJumps{0};

  void printDecision(llvm::raw_ostream &OS,
                     std::vector<IndirectCallPromotion::Callsite> &Targets,
                     unsigned N) const;

  std::vector<Callsite> getCallTargets(BinaryBasicBlock &BB,
                                       const MCInst &Inst) const;

  size_t canPromoteCallsite(const BinaryBasicBlock &BB, const MCInst &Inst,
                            const std::vector<Callsite> &Targets,
                            uint64_t NumCalls);

  void printCallsiteInfo(const BinaryBasicBlock &BB, const MCInst &Inst,
                         const std::vector<Callsite> &Targets, const size_t N,
                         uint64_t NumCalls) const;

  JumpTableInfoType maybeGetHotJumpTableTargets(BinaryBasicBlock &BB,
                                                MCInst &Inst,
                                                MCInst *&TargetFetchInst,
                                                const JumpTable *JT) const;

  SymTargetsType findCallTargetSymbols(std::vector<Callsite> &Targets,
                                       size_t &N, BinaryBasicBlock &BB,
                                       MCInst &Inst,
                                       MCInst *&TargetFetchInst) const;

  MethodInfoType maybeGetVtableSyms(BinaryBasicBlock &BB, MCInst &Inst,
                                    const SymTargetsType &SymTargets) const;

  std::vector<std::unique_ptr<BinaryBasicBlock>>
  rewriteCall(BinaryBasicBlock &IndCallBlock, const MCInst &CallInst,
              MCPlusBuilder::BlocksVectorTy &&ICPcode,
              const std::vector<MCInst *> &MethodFetchInsns) const;

  BinaryBasicBlock *fixCFG(BinaryBasicBlock &IndCallBlock,
                           const bool IsTailCall, const bool IsJumpTable,
                           BasicBlocksVector &&NewBBs,
                           const std::vector<Callsite> &Targets) const;

public:
  explicit IndirectCallPromotion(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "indirect-call-promotion"; }
  bool shouldPrint(const BinaryFunction &BF) const override {
    return BinaryFunctionPass::shouldPrint(BF) && Modified.count(&BF) > 0;
  }
  void runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
