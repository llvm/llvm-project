//===- bolt/Passes/LongJmp.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_LONGJMP_H
#define BOLT_PASSES_LONGJMP_H

#include "bolt/Passes/BinaryPasses.h"

namespace llvm {
namespace bolt {

/// LongJmp is veneer-insertion pass originally written for AArch64 that
/// compensates for its short-range branches, typically done during linking. We
/// pull this pass inside BOLT because here we can do a better job at stub
/// inserting by manipulating the CFG, something linkers can't do.
///
/// We iteratively repeat the following until no modification is done: we do a
/// tentative layout with the current function sizes; then we add stubs for
/// branches that we know are out of range or we expand smaller stubs (28-bit)
/// to a large one if necessary (32 or 64).
///
/// This expansion inserts the equivalent of "linker stubs", small
/// blocks of code that load a 64-bit address into a pre-allocated register and
//  then executes an unconditional indirect branch on this register. By using a
/// 64-bit range, we guarantee it can reach any code location.
///
class LongJmpPass : public BinaryFunctionPass {
  /// Used to implement stub grouping (reusing a stub from one function into
  /// another)
  using StubTy = std::pair<uint64_t, BinaryBasicBlock *>;
  using StubGroupTy = SmallVector<StubTy, 4>;
  using StubGroupsTy = DenseMap<const MCSymbol *, StubGroupTy>;
  StubGroupsTy HotStubGroups;
  StubGroupsTy ColdStubGroups;
  DenseMap<const MCSymbol *, BinaryBasicBlock *> SharedStubs;

  /// Stubs that are local to a function. This will be the primary lookup
  /// before resorting to stubs located in foreign functions.
  using StubMapTy = DenseMap<const BinaryFunction *, StubGroupsTy>;
  /// Used to quickly fetch stubs based on the target they jump to
  StubMapTy HotLocalStubs;
  StubMapTy ColdLocalStubs;

  /// Used to quickly identify whether a BB is a stub, sharded by function
  DenseMap<const BinaryFunction *, std::set<const BinaryBasicBlock *>> Stubs;

  using FuncAddressesMapTy = DenseMap<const BinaryFunction *, uint64_t>;
  /// Hold tentative addresses
  FuncAddressesMapTy HotAddresses;
  FuncAddressesMapTy ColdAddresses;
  DenseMap<const BinaryBasicBlock *, uint64_t> BBAddresses;

  /// Used to identify the stub size
  DenseMap<const BinaryBasicBlock *, int> StubBits;

  /// Stats about number of stubs inserted
  uint32_t NumHotStubs{0};
  uint32_t NumColdStubs{0};
  uint32_t NumSharedStubs{0};

  /// The shortest distance for any branch instruction on AArch64.
  static constexpr size_t ShortestJumpBits = 16;
  static constexpr size_t ShortestJumpSpan = 1ULL << (ShortestJumpBits - 1);

  /// The longest single-instruction branch.
  static constexpr size_t LongestJumpBits = 28;
  static constexpr size_t LongestJumpSpan = 1ULL << (LongestJumpBits - 1);

  /// Relax all internal function branches including those between fragments.
  /// Assume that fragments are placed in different sections but are within
  /// 128MB of each other.
  void relaxLocalBranches(BinaryFunction &BF);

  ///                 -- Layout estimation methods --
  /// Try to do layout before running the emitter, by looking at BinaryFunctions
  /// and MCInsts -- this is an estimation. To be correct for longjmp inserter
  /// purposes, we need to do a size worst-case estimation. Real layout is done
  /// by RewriteInstance::mapFileSections()
  void tentativeLayout(const BinaryContext &BC,
                       std::vector<BinaryFunction *> &SortedFunctions);
  uint64_t
  tentativeLayoutRelocMode(const BinaryContext &BC,
                           std::vector<BinaryFunction *> &SortedFunctions,
                           uint64_t DotAddress);
  uint64_t
  tentativeLayoutRelocColdPart(const BinaryContext &BC,
                               std::vector<BinaryFunction *> &SortedFunctions,
                               uint64_t DotAddress);
  void tentativeBBLayout(const BinaryFunction &Func);

  /// Update stubs addresses with their exact address after a round of stub
  /// insertion and layout estimation is done.
  void updateStubGroups();

  ///              -- Relaxation/stub insertion methods --
  /// Creates a  new stub jumping to \p TgtSym and updates bookkeeping about
  /// this stub using \p AtAddress as its initial location. This location is
  /// an approximation and will be later resolved to the exact location in
  /// a next iteration, in updateStubGroups.
  std::pair<std::unique_ptr<BinaryBasicBlock>, MCSymbol *>
  createNewStub(BinaryBasicBlock &SourceBB, const MCSymbol *TgtSym,
                bool TgtIsFunc, uint64_t AtAddress);

  /// Replace the target of call or conditional branch in \p Inst with a
  /// a stub that in turn will branch to the target (perform stub insertion).
  /// If a new stub was created, return it.
  std::unique_ptr<BinaryBasicBlock>
  replaceTargetWithStub(BinaryBasicBlock &BB, MCInst &Inst, uint64_t DotAddress,
                        uint64_t StubCreationAddress);

  /// Helper used to fetch the closest stub to \p Inst at \p DotAddress that
  /// is jumping to \p TgtSym. Returns nullptr if the closest stub is out of
  /// range or if it doesn't exist. The source of truth for stubs will be the
  /// map \p StubGroups, which can be either local stubs for a particular
  /// function that is very large and needs to group stubs, or can be global
  /// stubs if we are sharing stubs across functions.
  BinaryBasicBlock *lookupStubFromGroup(const StubGroupsTy &StubGroups,
                                        const BinaryFunction &Func,
                                        const MCInst &Inst,
                                        const MCSymbol *TgtSym,
                                        uint64_t DotAddress) const;

  /// Lookup closest stub from the global pool, meaning this can return a basic
  /// block from another function.
  BinaryBasicBlock *lookupGlobalStub(const BinaryBasicBlock &SourceBB,
                                     const MCInst &Inst, const MCSymbol *TgtSym,
                                     uint64_t DotAddress) const;

  /// Lookup closest stub local to \p Func.
  BinaryBasicBlock *lookupLocalStub(const BinaryBasicBlock &SourceBB,
                                    const MCInst &Inst, const MCSymbol *TgtSym,
                                    uint64_t DotAddress) const;

  /// Helper to identify whether \p Inst is branching to a stub
  bool usesStub(const BinaryFunction &Func, const MCInst &Inst) const;

  /// True if Inst is a branch that is out of range
  bool needsStub(const BinaryBasicBlock &BB, const MCInst &Inst,
                 uint64_t DotAddress) const;

  /// Expand the range of the stub in StubBB if necessary
  Error relaxStub(BinaryBasicBlock &StubBB, bool &Modified);

  /// Helper to resolve a symbol address according to our tentative layout
  uint64_t getSymbolAddress(const BinaryContext &BC, const MCSymbol *Target,
                            const BinaryBasicBlock *TgtBB) const;

  /// Relax function by adding necessary stubs or relaxing existing stubs
  Error relax(BinaryFunction &BF, bool &Modified);

public:
  /// BinaryPass public interface

  explicit LongJmpPass(const cl::opt<bool> &PrintPass)
      : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "long-jmp"; }

  Error runOnFunctions(BinaryContext &BC) override;
};
} // namespace bolt
} // namespace llvm

#endif
