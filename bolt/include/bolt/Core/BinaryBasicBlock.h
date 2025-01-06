//===- bolt/Core/BinaryBasicBlock.h - Low-level basic block -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sequence of MC/MCPlus instructions. Call/invoke does not terminate the block.
// CFI instructions are part of the instruction list with the initial CFI state
// defined at the beginning of the block.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_CORE_BINARY_BASIC_BLOCK_H
#define BOLT_CORE_BINARY_BASIC_BLOCK_H

#include "bolt/Core/FunctionLayout.h"
#include "bolt/Core/MCPlus.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <utility>

namespace llvm {
class MCCodeEmitter;

namespace bolt {

class BinaryFunction;
class JumpTable;

class BinaryBasicBlock {
public:
  /// Profile execution information for a given edge in CFG.
  ///
  /// If MispredictedCount equals COUNT_INFERRED, then we have a profile
  /// data for a fall-through edge with a Count representing an inferred
  /// execution count, i.e. the count we calculated internally, not the one
  /// coming from profile data.
  ///
  /// For all other values of MispredictedCount, Count represents the number of
  /// branch executions from a profile, and MispredictedCount is the number
  /// of times the branch was mispredicted according to this profile.
  struct BinaryBranchInfo {
    uint64_t Count;
    uint64_t MispredictedCount; /// number of branches mispredicted

    bool operator<(const BinaryBranchInfo &Other) const {
      return (Count < Other.Count) ||
             (Count == Other.Count &&
              MispredictedCount < Other.MispredictedCount);
    }
  };

  static constexpr uint32_t INVALID_OFFSET =
      std::numeric_limits<uint32_t>::max();

  using BranchInfoType = SmallVector<BinaryBranchInfo, 0>;

private:
  /// Vector of all instructions in the block.
  InstructionListType Instructions;

  /// CFG information.
  using EdgeListType = SmallVector<BinaryBasicBlock *, 0>;
  EdgeListType Predecessors;
  EdgeListType Successors;

  /// Each successor has a corresponding BranchInfo entry in the list.
  BranchInfoType BranchInfo;

  using ExceptionListType = SmallVector<BinaryBasicBlock *, 0>;

  /// List of blocks that this landing pad is handling.
  ExceptionListType Throwers;

  /// List of blocks that can catch exceptions thrown by code in this block.
  ExceptionListType LandingPads;

  /// Function that owns this basic block.
  BinaryFunction *Function;

  /// Label associated with the block.
  MCSymbol *Label{nullptr};

  /// [Begin, End) address range for this block in the output binary.
  std::pair<uint32_t, uint32_t> OutputAddressRange = {0, 0};

  /// Original offset range of the basic block in the function.
  std::pair<uint32_t, uint32_t> InputRange = {INVALID_OFFSET, INVALID_OFFSET};

  /// Map input offset (from function start) of an instruction to an output
  /// symbol. Enables writing BOLT address translation tables used for mapping
  /// control transfer in the output binary back to the original binary.
  using LocSymsTy = std::vector<std::pair<uint32_t, const MCSymbol *>>;
  std::unique_ptr<LocSymsTy> LocSyms;

  /// Alignment requirements for the block.
  uint32_t Alignment{1};

  /// Maximum number of bytes to use for alignment of the block.
  uint32_t AlignmentMaxBytes{0};

  /// Number of times this basic block was executed.
  uint64_t ExecutionCount{COUNT_NO_PROFILE};

  static constexpr unsigned InvalidIndex = ~0u;

  /// Index to BasicBlocks vector in BinaryFunction.
  unsigned Index{InvalidIndex};

  /// Index in the current layout.
  unsigned LayoutIndex{InvalidIndex};

  /// Number of pseudo instructions in this block.
  uint32_t NumPseudos{0};

  /// CFI state at the entry to this basic block.
  int32_t CFIState{-1};

  /// In cases where the parent function has been split, FragmentNum > 0 means
  /// this BB will be allocated in a fragment outside its parent function.
  FragmentNum Fragment;

  /// Indicates if the block could be outlined.
  bool CanOutline{true};

  /// Flag to indicate whether this block is valid or not.  Invalid
  /// blocks may contain out of date or incorrect information.
  bool IsValid{true};

  /// Last computed hash value.
  mutable uint64_t Hash{0};

private:
  BinaryBasicBlock() = delete;
  BinaryBasicBlock(const BinaryBasicBlock &) = delete;
  BinaryBasicBlock(const BinaryBasicBlock &&) = delete;
  BinaryBasicBlock &operator=(const BinaryBasicBlock &) = delete;
  BinaryBasicBlock &operator=(const BinaryBasicBlock &&) = delete;

  explicit BinaryBasicBlock(BinaryFunction *Function, MCSymbol *Label)
      : Function(Function), Label(Label) {
    assert(Function && "Function must be non-null");
  }

  // Exclusively managed by BinaryFunction.
  friend class BinaryFunction;
  friend bool operator<(const BinaryBasicBlock &LHS,
                        const BinaryBasicBlock &RHS);

  /// Assign new label to the basic block.
  void setLabel(MCSymbol *Symbol) { Label = Symbol; }

public:
  static constexpr uint64_t COUNT_INFERRED =
      std::numeric_limits<uint64_t>::max();
  static constexpr uint64_t COUNT_NO_PROFILE =
      std::numeric_limits<uint64_t>::max();

  // Instructions iterators.
  using iterator = InstructionListType::iterator;
  using const_iterator = InstructionListType::const_iterator;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  bool         empty()            const { assert(hasInstructions());
                                          return Instructions.empty(); }
  size_t       size()             const { assert(hasInstructions());
                                          return Instructions.size(); }
  MCInst       &front()                 { assert(hasInstructions());
                                          return Instructions.front();  }
  MCInst       &back()                  { assert(hasInstructions());
                                          return Instructions.back();   }
  const MCInst &front()           const { assert(hasInstructions());
                                          return Instructions.front();  }
  const MCInst &back()            const { assert(hasInstructions());
                                          return Instructions.back();   }

  iterator                begin()       { assert(hasInstructions());
                                          return Instructions.begin();  }
  const_iterator          begin() const { assert(hasInstructions());
                                          return Instructions.begin();  }
  iterator                end  ()       { assert(hasInstructions());
                                          return Instructions.end();    }
  const_iterator          end  () const { assert(hasInstructions());
                                          return Instructions.end();    }
  reverse_iterator       rbegin()       { assert(hasInstructions());
                                          return Instructions.rbegin(); }
  const_reverse_iterator rbegin() const { assert(hasInstructions());
                                          return Instructions.rbegin(); }
  reverse_iterator       rend  ()       { assert(hasInstructions());
                                          return Instructions.rend();   }
  const_reverse_iterator rend  () const { assert(hasInstructions());
                                          return Instructions.rend();   }

  // CFG iterators.
  using pred_iterator = EdgeListType::iterator;
  using const_pred_iterator = EdgeListType::const_iterator;
  using succ_iterator = EdgeListType::iterator;
  using const_succ_iterator = EdgeListType::const_iterator;
  using throw_iterator = decltype(Throwers)::iterator;
  using const_throw_iterator = decltype(Throwers)::const_iterator;
  using lp_iterator = decltype(LandingPads)::iterator;
  using const_lp_iterator = decltype(LandingPads)::const_iterator;

  using pred_reverse_iterator = std::reverse_iterator<pred_iterator>;
  using const_pred_reverse_iterator =
    std::reverse_iterator<const_pred_iterator>;
  using succ_reverse_iterator = std::reverse_iterator<succ_iterator>;
  using const_succ_reverse_iterator =
    std::reverse_iterator<const_succ_iterator>;

  pred_iterator        pred_begin()       { return Predecessors.begin(); }
  const_pred_iterator  pred_begin() const { return Predecessors.begin(); }
  pred_iterator        pred_end()         { return Predecessors.end();   }
  const_pred_iterator  pred_end()   const { return Predecessors.end();   }
  pred_reverse_iterator        pred_rbegin()
                                          { return Predecessors.rbegin();}
  const_pred_reverse_iterator  pred_rbegin() const
                                          { return Predecessors.rbegin();}
  pred_reverse_iterator        pred_rend()
                                          { return Predecessors.rend();  }
  const_pred_reverse_iterator  pred_rend()   const
                                          { return Predecessors.rend();  }
  size_t               pred_size()  const {
    return Predecessors.size();
  }
  bool                 pred_empty() const { return Predecessors.empty(); }

  succ_iterator        succ_begin()       { return Successors.begin();   }
  const_succ_iterator  succ_begin() const { return Successors.begin();   }
  succ_iterator        succ_end()         { return Successors.end();     }
  const_succ_iterator  succ_end()   const { return Successors.end();     }
  succ_reverse_iterator        succ_rbegin()
                                          { return Successors.rbegin();  }
  const_succ_reverse_iterator  succ_rbegin() const
                                          { return Successors.rbegin();  }
  succ_reverse_iterator        succ_rend()
                                          { return Successors.rend();    }
  const_succ_reverse_iterator  succ_rend()   const
                                          { return Successors.rend();    }
  size_t               succ_size()  const {
    return Successors.size();
  }
  bool                 succ_empty() const { return Successors.empty();   }

  throw_iterator        throw_begin()       { return Throwers.begin(); }
  const_throw_iterator  throw_begin() const { return Throwers.begin(); }
  throw_iterator        throw_end()         { return Throwers.end();   }
  const_throw_iterator  throw_end()   const { return Throwers.end();   }
  size_t                throw_size()  const {
    return Throwers.size();
  }
  bool                  throw_empty() const { return Throwers.empty(); }
  bool                  isLandingPad() const { return !Throwers.empty(); }

  lp_iterator        lp_begin()       { return LandingPads.begin();   }
  const_lp_iterator  lp_begin() const { return LandingPads.begin();   }
  lp_iterator        lp_end()         { return LandingPads.end();     }
  const_lp_iterator  lp_end()   const { return LandingPads.end();     }
  size_t             lp_size()  const {
    return LandingPads.size();
  }
  bool               lp_empty() const { return LandingPads.empty();   }

  inline iterator_range<iterator> instructions() {
    assert(hasInstructions());
    return iterator_range<iterator>(begin(), end());
  }
  inline iterator_range<const_iterator> instructions() const {
    assert(hasInstructions());
    return iterator_range<const_iterator>(begin(), end());
  }
  inline iterator_range<pred_iterator> predecessors() {
    assert(hasCFG());
    return iterator_range<pred_iterator>(pred_begin(), pred_end());
  }
  inline iterator_range<const_pred_iterator> predecessors() const {
    assert(hasCFG());
    return iterator_range<const_pred_iterator>(pred_begin(), pred_end());
  }
  inline iterator_range<succ_iterator> successors() {
    assert(hasCFG());
    return iterator_range<succ_iterator>(succ_begin(), succ_end());
  }
  inline iterator_range<const_succ_iterator> successors() const {
    assert(hasCFG());
    return iterator_range<const_succ_iterator>(succ_begin(), succ_end());
  }
  inline iterator_range<throw_iterator> throwers() {
    assert(hasCFG());
    return iterator_range<throw_iterator>(throw_begin(), throw_end());
  }
  inline iterator_range<const_throw_iterator> throwers() const {
    assert(hasCFG());
    return iterator_range<const_throw_iterator>(throw_begin(), throw_end());
  }
  inline iterator_range<lp_iterator> landing_pads() {
    assert(hasCFG());
    return iterator_range<lp_iterator>(lp_begin(), lp_end());
  }
  inline iterator_range<const_lp_iterator> landing_pads() const {
    assert(hasCFG());
    return iterator_range<const_lp_iterator>(lp_begin(), lp_end());
  }

  // BranchInfo iterators.
  using branch_info_iterator = BranchInfoType::iterator;
  using const_branch_info_iterator = BranchInfoType::const_iterator;
  using branch_info_reverse_iterator =
      std::reverse_iterator<branch_info_iterator>;
  using const_branch_info_reverse_iterator =
      std::reverse_iterator<const_branch_info_iterator>;

  branch_info_iterator branch_info_begin() { return BranchInfo.begin(); }
  branch_info_iterator branch_info_end() { return BranchInfo.end(); }
  const_branch_info_iterator branch_info_begin() const {
    return BranchInfo.begin();
  }
  const_branch_info_iterator branch_info_end() const {
    return BranchInfo.end();
  }
  branch_info_reverse_iterator branch_info_rbegin() {
    return BranchInfo.rbegin();
  }
  branch_info_reverse_iterator branch_info_rend() { return BranchInfo.rend(); }
  const_branch_info_reverse_iterator branch_info_rbegin() const {
    return BranchInfo.rbegin();
  }
  const_branch_info_reverse_iterator branch_info_rend() const {
    return BranchInfo.rend();
  }

  size_t branch_info_size() const { return BranchInfo.size(); }
  bool branch_info_empty() const { return BranchInfo.empty(); }

  inline iterator_range<branch_info_iterator> branch_info() {
    return iterator_range<branch_info_iterator>(BranchInfo.begin(),
                                                BranchInfo.end());
  }
  inline iterator_range<const_branch_info_iterator> branch_info() const {
    return iterator_range<const_branch_info_iterator>(BranchInfo.begin(),
                                                      BranchInfo.end());
  }

  /// Get instruction at given index.
  MCInst &getInstructionAtIndex(unsigned Index) { return Instructions[Index]; }

  const MCInst &getInstructionAtIndex(unsigned Index) const {
    return Instructions[Index];
  }

  /// Return symbol marking the start of this basic block.
  MCSymbol *getLabel() { return Label; }

  /// Return symbol marking the start of this basic block (const version).
  const MCSymbol *getLabel() const { return Label; }

  /// Get successor with given \p Label if \p Label != nullptr.
  /// Returns nullptr if no such successor is found.
  /// If the \p Label == nullptr and the block has only one successor then
  /// return the successor.
  BinaryBasicBlock *getSuccessor(const MCSymbol *Label = nullptr) const;

  /// Return the related branch info as well as the successor.
  BinaryBasicBlock *getSuccessor(const MCSymbol *Label,
                                 BinaryBranchInfo &BI) const;

  /// If the basic block ends with a conditional branch (possibly followed by
  /// an unconditional branch) and thus has 2 successors, return a successor
  /// corresponding to a jump condition which could be true or false.
  /// Return nullptr if the basic block does not have a conditional jump.
  BinaryBasicBlock *getConditionalSuccessor(bool Condition) {
    if (succ_size() != 2)
      return nullptr;
    return Successors[Condition == true ? 0 : 1];
  }

  const BinaryBasicBlock *getConditionalSuccessor(bool Condition) const {
    return const_cast<BinaryBasicBlock *>(this)->getConditionalSuccessor(
        Condition);
  }

  /// Find the fallthrough successor for a block, or nullptr if there is
  /// none.
  BinaryBasicBlock *getFallthrough() {
    if (succ_size() == 2)
      return getConditionalSuccessor(false);
    else
      return getSuccessor();
  }

  const BinaryBasicBlock *getFallthrough() const {
    return const_cast<BinaryBasicBlock *>(this)->getFallthrough();
  }

  /// Return branch info corresponding to a taken branch.
  const BinaryBranchInfo &getTakenBranchInfo() const {
    assert(BranchInfo.size() == 2 &&
           "could only be called for blocks with 2 successors");
    return BranchInfo[0];
  };

  /// Return branch info corresponding to a fall-through branch.
  const BinaryBranchInfo &getFallthroughBranchInfo() const {
    assert(BranchInfo.size() == 2 &&
           "could only be called for blocks with 2 successors");
    return BranchInfo[1];
  };

  /// Return branch info corresponding to an edge going to \p Succ basic block.
  BinaryBranchInfo &getBranchInfo(const BinaryBasicBlock &Succ);

  /// Return branch info corresponding to an edge going to \p Succ basic block.
  const BinaryBranchInfo &getBranchInfo(const BinaryBasicBlock &Succ) const;

  /// Set branch information for the outgoing edge to block \p Succ.
  void setSuccessorBranchInfo(const BinaryBasicBlock &Succ, uint64_t Count,
                              uint64_t MispredictedCount) {
    BinaryBranchInfo &BI = getBranchInfo(Succ);
    BI.Count = Count;
    BI.MispredictedCount = MispredictedCount;
  }

  /// Try to compute the taken and misprediction frequencies for the given
  /// successor.  The result is an error if no information can be found.
  ErrorOr<std::pair<double, double>>
  getBranchStats(const BinaryBasicBlock *Succ) const;

  /// If the basic block ends with a conditional branch (possibly followed by
  /// an unconditional branch) and thus has 2 successor, reverse the order of
  /// its successors in CFG, update branch info, and return true. If the basic
  /// block does not have 2 successors return false.
  bool swapConditionalSuccessors();

  /// Add an instruction with unconditional control transfer to \p Successor
  /// basic block to the end of this basic block.
  void addBranchInstruction(const BinaryBasicBlock *Successor);

  /// Add an instruction with tail call control transfer to \p Target
  /// to the end of this basic block.
  void addTailCallInstruction(const MCSymbol *Target);

  /// Return the number of call instructions in this basic block.
  uint32_t getNumCalls() const;

  /// Get landing pad with given label. Returns nullptr if no such
  /// landing pad is found.
  BinaryBasicBlock *getLandingPad(const MCSymbol *Label) const;

  /// Return local name for the block.
  StringRef getName() const { return Label->getName(); }

  /// Add instruction at the end of this basic block.
  /// Returns iterator pointing to the inserted instruction.
  iterator addInstruction(MCInst &&Inst) {
    adjustNumPseudos(Inst, 1);
    Instructions.emplace_back(Inst);
    return std::prev(Instructions.end());
  }

  /// Add instruction at the end of this basic block.
  /// Returns iterator pointing to the inserted instruction.
  iterator addInstruction(const MCInst &Inst) {
    adjustNumPseudos(Inst, 1);
    Instructions.push_back(Inst);
    return std::prev(Instructions.end());
  }

  /// Add a range of instructions to the end of this basic block.
  template <typename Itr> void addInstructions(Itr Begin, Itr End) {
    while (Begin != End)
      addInstruction(*Begin++);
  }

  /// Add a range of instructions to the end of this basic block.
  template <typename RangeTy> void addInstructions(RangeTy R) {
    for (auto &I : R)
      addInstruction(I);
  }

  /// Add instruction before Pos in this basic block.
  template <typename Itr> Itr insertPseudoInstr(Itr Pos, MCInst &Instr) {
    ++NumPseudos;
    return Instructions.insert(Pos, Instr);
  }

  /// Return the number of pseudo instructions in the basic block.
  uint32_t getNumPseudos() const;

  /// Return the number of emitted instructions for this basic block.
  uint32_t getNumNonPseudos() const { return size() - getNumPseudos(); }

  /// Return iterator to the first non-pseudo instruction or end()
  /// if no such instruction was found.
  iterator getFirstNonPseudo();

  /// Return a pointer to the first non-pseudo instruction in this basic
  /// block.  Returns nullptr if none exists.
  MCInst *getFirstNonPseudoInstr() {
    auto II = getFirstNonPseudo();
    return II == Instructions.end() ? nullptr : &*II;
  }

  /// Return reverse iterator to the last non-pseudo instruction or rend()
  /// if no such instruction was found.
  reverse_iterator getLastNonPseudo();
  const_reverse_iterator getLastNonPseudo() const {
    return const_cast<BinaryBasicBlock *>(this)->getLastNonPseudo();
  }

  /// Return a pointer to the last non-pseudo instruction in this basic
  /// block.  Returns nullptr if none exists.
  MCInst *getLastNonPseudoInstr() {
    auto RII = getLastNonPseudo();
    return RII == Instructions.rend() ? nullptr : &*RII;
  }
  const MCInst *getLastNonPseudoInstr() const {
    auto RII = getLastNonPseudo();
    return RII == Instructions.rend() ? nullptr : &*RII;
  }

  /// Set CFI state at entry to this basic block.
  void setCFIState(int32_t NewCFIState) {
    assert((CFIState == -1 || NewCFIState == CFIState) &&
           "unexpected change of CFI state for basic block");
    CFIState = NewCFIState;
  }

  /// Return CFI state (expected) at entry of this basic block.
  int32_t getCFIState() const {
    assert(CFIState >= 0 && "unknown CFI state");
    return CFIState;
  }

  /// Calculate and return CFI state right before instruction \p Instr in
  /// this basic block. If \p Instr is nullptr then return the state at
  /// the end of the basic block.
  int32_t getCFIStateAtInstr(const MCInst *Instr) const;

  /// Calculate and return CFI state after execution of this basic block.
  /// The state depends on CFI state at entry and CFI instructions inside the
  /// basic block.
  int32_t getCFIStateAtExit() const { return getCFIStateAtInstr(nullptr); }

  /// Set minimum alignment for the basic block.
  void setAlignment(uint32_t Align) { Alignment = Align; }

  /// Set alignment of the block based on the alignment of its offset.
  void setDerivedAlignment() {
    const uint64_t DerivedAlignment = getOffset() & (1 + ~getOffset());
    Alignment = std::min(DerivedAlignment, uint64_t(32));
  }

  /// Return required alignment for the block.
  Align getAlign() const { return Align(Alignment); }
  uint32_t getAlignment() const { return Alignment; }

  /// Set the maximum number of bytes to use for the block alignment.
  void setAlignmentMaxBytes(uint32_t Value) { AlignmentMaxBytes = Value; }

  /// Return the maximum number of bytes to use for the block alignment.
  uint32_t getAlignmentMaxBytes() const { return AlignmentMaxBytes; }

  /// Adds block to successor list, and also updates predecessor list for
  /// successor block.
  /// Set branch info for this path.
  void addSuccessor(BinaryBasicBlock *Succ, uint64_t Count = 0,
                    uint64_t MispredictedCount = 0);

  void addSuccessor(BinaryBasicBlock *Succ, const BinaryBranchInfo &BI) {
    addSuccessor(Succ, BI.Count, BI.MispredictedCount);
  }

  /// Add a range of successors.
  template <typename Itr> void addSuccessors(Itr Begin, Itr End) {
    while (Begin != End)
      addSuccessor(*Begin++);
  }

  /// Add a range of successors with branch info.
  template <typename Itr, typename BrItr>
  void addSuccessors(Itr Begin, Itr End, BrItr BrBegin, BrItr BrEnd) {
    assert(std::distance(Begin, End) == std::distance(BrBegin, BrEnd));
    while (Begin != End)
      addSuccessor(*Begin++, *BrBegin++);
  }

  /// Replace Succ with NewSucc.  This routine is helpful for preserving
  /// the order of conditional successors when editing the CFG.
  void replaceSuccessor(BinaryBasicBlock *Succ, BinaryBasicBlock *NewSucc,
                        uint64_t Count = 0, uint64_t MispredictedCount = 0);

  /// Move all of this block's successors to a new block, and set the
  /// execution count of this new block with our execution count. This is
  /// useful when splitting a block in two.
  void moveAllSuccessorsTo(BinaryBasicBlock *New) {
    New->addSuccessors(successors().begin(), successors().end(),
                       branch_info_begin(), branch_info_end());
    removeAllSuccessors();

    // Update the execution count on the new block.
    New->setExecutionCount(getExecutionCount());
  }

  /// Remove /p Succ basic block from the list of successors. Update the
  /// list of predecessors of /p Succ and update branch info.
  void removeSuccessor(BinaryBasicBlock *Succ);

  /// Remove all successors of the basic block, and remove the block
  /// from respective lists of predecessors.
  void removeAllSuccessors();

  /// Remove useless duplicate successors.  When the conditional
  /// successor is the same as the unconditional successor, we can
  /// remove the conditional successor and branch instruction.
  void removeDuplicateConditionalSuccessor(MCInst *CondBranch);

  /// Update successors of the basic block based on the jump table instruction.
  /// The block must end with a jump table instruction.
  void updateJumpTableSuccessors();

  /// Test if BB is a predecessor of this block.
  bool isPredecessor(const BinaryBasicBlock *BB) const {
    return llvm::is_contained(Predecessors, BB);
  }

  /// Test if BB is a successor of this block.
  bool isSuccessor(const BinaryBasicBlock *BB) const {
    return llvm::is_contained(Successors, BB);
  }

  /// Test if this BB has a valid execution count.
  bool hasProfile() const { return ExecutionCount != COUNT_NO_PROFILE; }

  /// Return the information about the number of times this basic block was
  /// executed.
  ///
  /// Return COUNT_NO_PROFILE if there's no profile info.
  uint64_t getExecutionCount() const { return ExecutionCount; }

  /// Return the execution count for blocks with known profile.
  /// Return 0 if the block has no profile.
  uint64_t getKnownExecutionCount() const {
    return !hasProfile() ? 0 : ExecutionCount;
  }

  /// Set the execution count for this block.
  void setExecutionCount(uint64_t Count) { ExecutionCount = Count; }

  /// Apply a given \p Ratio to the profile information of this basic block.
  void adjustExecutionCount(double Ratio);

  /// Return true if the basic block is an entry point into the function
  /// (either primary or secondary).
  bool isEntryPoint() const;

  bool isValid() const { return IsValid; }

  void markValid(const bool Valid) { IsValid = Valid; }

  FragmentNum getFragmentNum() const { return Fragment; }

  void setFragmentNum(const FragmentNum Value) { Fragment = Value; }

  bool isSplit() const { return Fragment != FragmentNum::main(); }

  bool isCold() const {
    assert(Fragment.get() < 2 &&
           "Function is split into more than two (hot/cold)-fragments");
    return isSplit();
  }

  void setIsCold(const bool Flag) {
    Fragment = Flag ? FragmentNum::cold() : FragmentNum::main();
  }

  /// Return true if the block can be outlined. At the moment we disallow
  /// outlining of blocks that can potentially throw exceptions or are
  /// the beginning of a landing pad. The entry basic block also can
  /// never be outlined.
  bool canOutline() const { return CanOutline; }

  void setCanOutline(const bool Flag) { CanOutline = Flag; }

  /// Erase pseudo instruction at a given iterator.
  /// Return iterator following the removed instruction.
  iterator erasePseudoInstruction(iterator II) {
    --NumPseudos;
    return Instructions.erase(II);
  }

  /// Erase non-pseudo instruction at a given iterator \p II.
  /// Return iterator following the removed instruction.
  iterator eraseInstruction(iterator II) {
    adjustNumPseudos(*II, -1);
    return Instructions.erase(II);
  }

  /// Erase non-pseudo instruction at a given \p Index
  void eraseInstructionAtIndex(unsigned Index) {
    eraseInstruction(Instructions.begin() + Index);
  }

  /// Erase instructions in the specified range.
  template <typename ItrType>
  void eraseInstructions(ItrType Begin, ItrType End) {
    while (End > Begin)
      eraseInstruction(findInstruction(*--End));
  }

  /// Erase all instructions.
  void clear() {
    Instructions.clear();
    NumPseudos = 0;
  }

  /// Retrieve iterator for \p Inst or return end iterator if instruction is not
  /// from this basic block.
  decltype(Instructions)::iterator findInstruction(const MCInst *Inst) {
    if (Instructions.empty())
      return Instructions.end();
    size_t Index = Inst - &Instructions[0];
    return Index >= Instructions.size() ? Instructions.end()
                                        : Instructions.begin() + Index;
  }

  /// Replace instruction referenced by iterator \II with a sequence of
  /// instructions defined by [\p Begin, \p End] range.
  ///
  /// Return iterator pointing to the first inserted instruction.
  template <typename Itr>
  iterator replaceInstruction(iterator II, Itr Begin, Itr End) {
    adjustNumPseudos(*II, -1);
    adjustNumPseudos(Begin, End, 1);

    auto I = II - Instructions.begin();
    Instructions.insert(Instructions.erase(II), Begin, End);
    return I + Instructions.begin();
  }

  iterator replaceInstruction(iterator II,
                              const InstructionListType &Replacement) {
    return replaceInstruction(II, Replacement.begin(), Replacement.end());
  }

  /// Insert \p NewInst before \p At, which must be an existing instruction in
  /// this BB. Return iterator pointing to the newly inserted instruction.
  iterator insertInstruction(iterator At, MCInst &&NewInst) {
    adjustNumPseudos(NewInst, 1);
    return Instructions.emplace(At, std::move(NewInst));
  }

  iterator insertInstruction(iterator At, MCInst &NewInst) {
    adjustNumPseudos(NewInst, 1);
    return Instructions.insert(At, NewInst);
  }

  /// Helper to retrieve any terminators in \p BB before \p Pos. This is used
  /// to skip CFI instructions and to retrieve the first terminator instruction
  /// in basic blocks with two terminators (conditional jump and unconditional
  /// jump).
  MCInst *getTerminatorBefore(MCInst *Pos);

  /// Used to identify whether an instruction is before a terminator and whether
  /// moving it to the end of the BB would render it dead code.
  bool hasTerminatorAfter(MCInst *Pos);

  /// Split apart the instructions in this basic block starting at Inst.
  /// The instructions following Inst are removed and returned in a vector.
  InstructionListType splitInstructions(const MCInst *Inst) {
    InstructionListType SplitInst;

    assert(!Instructions.empty());
    while (&Instructions.back() != Inst) {
      SplitInst.push_back(Instructions.back());
      Instructions.pop_back();
    }
    std::reverse(SplitInst.begin(), SplitInst.end());
    NumPseudos = 0;
    adjustNumPseudos(Instructions.begin(), Instructions.end(), 1);
    return SplitInst;
  }

  /// Split basic block at the instruction pointed to by II.
  /// All iterators pointing after II get invalidated.
  ///
  /// Return the new basic block that starts with the instruction
  /// at the split point.
  BinaryBasicBlock *splitAt(iterator II);

  /// Set start offset of this basic block in the input binary.
  void setOffset(uint32_t Offset) { InputRange.first = Offset; };

  /// Sets address of the basic block in the output.
  void setOutputStartAddress(uint64_t Address) {
    OutputAddressRange.first = Address;
  }

  /// Sets address past the end of the basic block in the output.
  void setOutputEndAddress(uint64_t Address) {
    OutputAddressRange.second = Address;
  }

  /// Gets the memory address range of this BB in the input binary.
  std::pair<uint64_t, uint64_t> getInputAddressRange() const {
    return InputRange;
  }

  /// Gets the memory address range of this BB in the output binary.
  std::pair<uint64_t, uint64_t> getOutputAddressRange() const {
    return OutputAddressRange;
  }

  uint64_t getOutputStartAddress() const { return OutputAddressRange.first; }
  uint64_t getOutputEndAddress() const { return OutputAddressRange.second; }

  bool hasLocSyms() const { return LocSyms != nullptr; }

  /// Return mapping of input offsets to symbols in the output.
  LocSymsTy &getLocSyms() {
    return LocSyms ? *LocSyms : *(LocSyms = std::make_unique<LocSymsTy>());
  }

  /// Return mapping of input offsets to symbols in the output.
  const LocSymsTy &getLocSyms() const {
    return const_cast<BinaryBasicBlock *>(this)->getLocSyms();
  }

  /// Return size of the basic block in the output binary.
  uint64_t getOutputSize() const {
    return OutputAddressRange.second - OutputAddressRange.first;
  }

  BinaryFunction *getFunction() const { return Function; }

  /// Analyze and interpret the terminators of this basic block. TBB must be
  /// initialized with the original fall-through for this BB.
  bool analyzeBranch(const MCSymbol *&TBB, const MCSymbol *&FBB,
                     MCInst *&CondBranch, MCInst *&UncondBranch);

  /// Printer required for printing dominator trees.
  void printAsOperand(raw_ostream &OS, bool PrintType = true) {
    if (PrintType)
      OS << "basic block ";
    OS << getName();
  }

  /// A simple dump function for debugging.
  void dump() const;

  /// Validate successor invariants for this BB.
  bool validateSuccessorInvariants();

  /// Return offset of the basic block from the function start on input.
  uint32_t getInputOffset() const { return InputRange.first; }

  /// Return offset from the function start to location immediately past
  /// the end of the basic block.
  uint32_t getEndOffset() const { return InputRange.second; }

  /// Return size of the basic block on input.
  uint32_t getOriginalSize() const {
    return InputRange.second - InputRange.first;
  }

  /// Returns an estimate of size of basic block during run time optionally
  /// using a user-supplied emitter for lock-free multi-thread work.
  /// MCCodeEmitter is not thread safe and each thread should operate with its
  /// own copy of it.
  uint64_t estimateSize(const MCCodeEmitter *Emitter = nullptr) const;

  /// Return index in the current layout. The user is responsible for
  /// making sure the indices are up to date,
  /// e.g. by calling BinaryFunction::updateLayoutIndices();
  unsigned getLayoutIndex() const {
    assert(isValid());
    return LayoutIndex;
  }

  /// Set layout index. To be used by BinaryFunction.
  void setLayoutIndex(unsigned Index) { LayoutIndex = Index; }

  /// Needed by graph traits.
  BinaryFunction *getParent() const { return getFunction(); }

  /// Return true if the containing function is in CFG state.
  bool hasCFG() const;

  /// Return true if the containing function is in a state with instructions.
  bool hasInstructions() const;

  /// Return offset of the basic block from the function start.
  uint32_t getOffset() const { return InputRange.first; }

  /// Get the index of this basic block.
  unsigned getIndex() const {
    assert(isValid());
    return Index;
  }

  /// Return jump table if the block contains a jump table instruction or
  /// nullptr otherwise.
  const JumpTable *getJumpTable() const;

  /// Check if the block has a jump table instruction.
  bool hasJumpTable() const { return getJumpTable() != nullptr; }

  /// Returns the last computed hash value of the block.
  uint64_t getHash() const { return Hash; }

private:
  void adjustNumPseudos(const MCInst &Inst, int Sign);

  template <typename Itr> void adjustNumPseudos(Itr Begin, Itr End, int Sign) {
    while (Begin != End)
      adjustNumPseudos(*Begin++, Sign);
  }

  /// Adds predecessor to the BB. Most likely you don't need to call this.
  void addPredecessor(BinaryBasicBlock *Pred);

  /// Remove predecessor of the basic block. Don't use directly, instead
  /// use removeSuccessor() function.
  /// If \p Multiple is set to true, it will remove all predecessors that
  /// are equal to \p Pred. Otherwise, the first instance of \p Pred found
  /// will be removed. This only matters in awkward, redundant CFGs.
  void removePredecessor(BinaryBasicBlock *Pred, bool Multiple = true);

  /// Set end offset of this basic block.
  void setEndOffset(uint32_t Offset) { InputRange.second = Offset; }

  /// Set the index of this basic block.
  void setIndex(unsigned I) { Index = I; }

  /// Sets the hash value of the basic block.
  void setHash(uint64_t Value) const { Hash = Value; }

  template <typename T> void clearList(T &List) {
    T TempList;
    TempList.swap(List);
  }

  /// Release memory taken by CFG edges and instructions.
  void releaseCFG() {
    clearList(Predecessors);
    clearList(Successors);
    clearList(Throwers);
    clearList(LandingPads);
    clearList(BranchInfo);
    clearList(Instructions);
  }
};

#if defined(LLVM_ON_UNIX)
/// Keep the size of the BinaryBasicBlock within a reasonable size class
/// (jemalloc bucket) on Linux
static_assert(sizeof(BinaryBasicBlock) <= 256);
#endif

bool operator<(const BinaryBasicBlock &LHS, const BinaryBasicBlock &RHS);

} // namespace bolt

// GraphTraits specializations for basic block graphs (CFGs)
template <> struct GraphTraits<bolt::BinaryBasicBlock *> {
  using NodeRef = bolt::BinaryBasicBlock *;
  using ChildIteratorType = bolt::BinaryBasicBlock::succ_iterator;

  static NodeRef getEntryNode(bolt::BinaryBasicBlock *BB) { return BB; }
  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->succ_begin();
  }
  static inline ChildIteratorType child_end(NodeRef N) { return N->succ_end(); }
};

template <> struct GraphTraits<const bolt::BinaryBasicBlock *> {
  using NodeRef = const bolt::BinaryBasicBlock *;
  using ChildIteratorType = bolt::BinaryBasicBlock::const_succ_iterator;

  static NodeRef getEntryNode(const bolt::BinaryBasicBlock *BB) { return BB; }
  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->succ_begin();
  }
  static inline ChildIteratorType child_end(NodeRef N) { return N->succ_end(); }
};

template <> struct GraphTraits<Inverse<bolt::BinaryBasicBlock *>> {
  using NodeRef = bolt::BinaryBasicBlock *;
  using ChildIteratorType = bolt::BinaryBasicBlock::pred_iterator;
  static NodeRef getEntryNode(Inverse<bolt::BinaryBasicBlock *> G) {
    return G.Graph;
  }
  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->pred_begin();
  }
  static inline ChildIteratorType child_end(NodeRef N) { return N->pred_end(); }
};

template <> struct GraphTraits<Inverse<const bolt::BinaryBasicBlock *>> {
  using NodeRef = const bolt::BinaryBasicBlock *;
  using ChildIteratorType = bolt::BinaryBasicBlock::const_pred_iterator;
  static NodeRef getEntryNode(Inverse<const bolt::BinaryBasicBlock *> G) {
    return G.Graph;
  }
  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->pred_begin();
  }
  static inline ChildIteratorType child_end(NodeRef N) { return N->pred_end(); }
};

} // namespace llvm

#endif
