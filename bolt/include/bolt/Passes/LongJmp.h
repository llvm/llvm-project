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

class BranchLivenessInfo;

/// LongJmp is veneer-insertion pass originally written for AArch64 that
/// compensates for its short-range branches, typically done during linking. We
/// pull this pass inside BOLT because here we can do a better job at stub
/// inserting by manipulating the CFG, something linkers can't do.
///
/// We iteratively repeat the following until no modification is done: we
/// compute the layout with the current function sizes; then we add stubs for
/// branches that we know are out of range or we expand smaller stubs (28-bit)
/// to a large one if necessary (32 or 64).
///
/// This expansion inserts the equivalent of "linker stubs": small blocks of
/// code that load a 64-bit address into a pre-allocated register and then
/// execute an unconditional indirect branch through that register. By using a
/// 64-bit range, we guarantee that they can reach any code location.
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
  /// Main-fragment start addresses for the current layout iteration.
  FuncAddressesMapTy HotAddresses;

  /// Basic-block start addresses for the current layout iteration.
  DenseMap<const BinaryBasicBlock *, uint64_t> BBAddresses;

  /// Used to identify the stub size
  DenseMap<const BinaryBasicBlock *, int> StubBits;

  /// Stats about number of stubs inserted
  uint32_t NumHotStubs{0};
  uint32_t NumColdStubs{0};
  uint32_t NumSharedStubs{0};

  /// The shortest distance for any branch instruction on AArch64.
  static constexpr size_t ShortestJumpBits = 11;
  static constexpr size_t ShortestJumpSpan = 1ULL << (ShortestJumpBits - 1);

  /// The longest single-instruction branch.
  static constexpr size_t LongestJumpBits = 28;
  static constexpr size_t LongestJumpSpan = 1ULL << (LongestJumpBits - 1);

  /// Relax all internal function branches including those between fragments.
  /// Assume that fragments are placed in different sections but are within
  /// 128MB of each other. Return false and report an error if a branch cannot
  /// be relaxed.
  bool relaxLocalBranches(BinaryFunction &BF,
                          const BranchLivenessInfo *BLI = nullptr);

  /// A group of functions that are located within the longest direct
  /// branch/call instruction distance. Functions within the cluster do not
  /// require a thunk for calls in the same cluster. The cluster may include
  /// a set of thunks for covering calls to functions outside.
  struct FunctionCluster {
    /// All functions in this cluster.
    DenseSet<BinaryFunction *> Functions;

    /// Symbols corresponding to entry points of functions that this cluster
    /// calls. Note that it excludes all functions in the cluster itself.
    DenseSet<const MCSymbol *> Callees;

    /// Estimated size of the cluster in bytes.
    uint64_t Size{0};

    /// The index of the last function in the cluster. Used as an insertion
    /// point for adding thunks to the output function list.
    size_t LastFunctionIndex = -1;

    /// When placing hot code at the end of the binary, track the first function
    /// for insertion purposes.
    size_t FirstFunctionIndex = -1;

    /// Thunks located at the end of this cluster.
    BinaryFunctionListType ThunkList;

    /// Thunks used by this cluster. Some could be in a ThunkList of the
    /// preceding cluster.
    ///
    /// <Function Symbol> -> <Thunk Function>.
    DenseMap<const MCSymbol *, BinaryFunction *> Thunks;
  };

  /// Maximum size of combined regular functions in the cluster. Note that it's
  /// less than 128MB, because the size of the cluster plus its thunks should be
  /// less than 128MB.
  static constexpr uint64_t MaxClusterSize = 125 * 1024 * 1024;

  /// Relax calls using function cluster approach.
  void relaxCalls(BinaryContext &BC);

  /// Identifies a function fragment and its owning function.
  struct FunctionFragmentPlacement {
    /// Function that owns the fragment.
    const BinaryFunction *Func;

    /// Number identifying the fragment within the function layout.
    FragmentNum Fragment;
  };

  /// Describes the placement of one emitted code section.
  struct SectionPlacement {
    /// Output section name returned by BinaryFunction::getCodeSectionName().
    SmallString<32> Name;

    /// Fragments in BinaryEmitter::emitFunctions() emission order.
    SmallVector<FunctionFragmentPlacement, 0> Fragments;

    /// Maximum alignment required by the section and its contents.
    uint64_t Alignment;
  };

  /// Code sections whose addresses are determined during final mapping, and
  /// their fragments. In relocation mode, sections are sorted like
  /// RewriteInstance::getCodeSections() before allocation. In non-relocation
  /// mode, these are the non-fixed injected sections in emission order.
  SmallVector<SectionPlacement, 4> Sections;

  /// Update \p CurrentAlignment with the requirements added while
  /// BinaryEmitter emits \p FF. The non-relocation path is used only for
  /// non-fixed injected sections.
  uint64_t updateSectionAlignment(const BinaryContext &BC,
                                  const BinaryFunction &Func,
                                  const FunctionFragment &FF,
                                  uint64_t CurrentAlignment) const;

  /// Assign \p FF to its output section, creating its placement if needed.
  void assignFunctionFragmentToSection(const BinaryContext &BC,
                                       const BinaryFunction &Func,
                                       const FunctionFragment &FF);

  /// Assign fragments to output sections in BinaryEmitter::emitFunctions()
  /// order. In relocation mode, collect all emitted non-fixed fragments. In
  /// non-relocation mode, collect only non-fixed injected functions. Functions
  /// mapped independently by RewriteInstance are excluded.
  void assignFunctionsToSections(const BinaryContext &BC,
                                 const BinaryFunctionListType &SortedFunctions);

  /// Mirror the code-section contents emitted by
  /// BinaryEmitter::emitFunctionBody(). Labels, CFI and debug directives do not
  /// advance the address and are omitted. If \p RecordAddresses is false, only
  /// calculate the ending address.
  uint64_t layoutFunctionBody(const BinaryContext &BC,
                              const BinaryFunction &Func,
                              const FunctionFragment &FF, uint64_t DotAddress,
                              bool RecordAddresses = true);

  /// Mirror the code-section contents emitted by BinaryEmitter::emitFunction().
  /// \p Func must pass shouldEmitFunctionFragment(). If \p RecordAddresses is
  /// false, only calculate the ending address.
  uint64_t layoutFunctionFragment(const BinaryContext &BC,
                                  const BinaryFunction &Func,
                                  const FunctionFragment &FF,
                                  uint64_t DotAddress,
                                  bool RecordAddresses = true);

  /// Lay out the fragments in \p Section in emission order and return the first
  /// address after the section. If \p RecordAddresses is false, do not update
  /// the function or basic-block address maps.
  uint64_t layoutSection(const BinaryContext &BC,
                         const SectionPlacement &Section, uint64_t DotAddress,
                         bool RecordAddresses = true);

  /// Lay out Sections toward increasing addresses and return the first address
  /// after them. This mirrors allocateAt() in relocation mode and the injected
  /// section allocation in mapCodeSectionsInPlace() otherwise.
  uint64_t layoutSectionsForward(const BinaryContext &BC, uint64_t DotAddress);

  /// Mirror the relocation-mode allocateBefore() helper in
  /// RewriteInstance::mapCodeSections(). Calculate section sizes while
  /// allocating before \p DotAddress. Return false if subtraction would
  /// underflow or alignment would place a section before the start of old
  /// .text (BC.OldTextSectionAddress); otherwise record the layout and return
  /// true.
  bool layoutSectionsBackward(const BinaryContext &BC, uint64_t DotAddress);

  /// Lay out section-mapped and independently placed functions according to
  /// the mapping rules for the current relocation mode.
  void layoutFunctions(const BinaryContext &BC,
                       const BinaryFunctionListType &SortedFunctions);

  /// Compute the code layout for the current LongJmp iteration by mirroring
  /// BinaryEmitter emission and RewriteInstance section mapping.
  void layout(const BinaryContext &BC,
              const BinaryFunctionListType &SortedFunctions);

  /// Update stub addresses after computing the current layout.
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

  /// Helper to resolve a symbol address according to our computed layout.
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
