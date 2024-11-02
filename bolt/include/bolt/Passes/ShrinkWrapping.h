//===- bolt/Passes/ShrinkWrapping.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_SHRINKWRAPPING_H
#define BOLT_PASSES_SHRINKWRAPPING_H

#include "bolt/Passes/FrameAnalysis.h"
#include "llvm/MC/MCRegisterInfo.h"
#include <atomic>

namespace llvm {
namespace bolt {
class DataflowInfoManager;

/// Encapsulates logic required to analyze a binary function and detect which
/// registers are being saved as callee-saved, where are these saves and where
/// are the points where their original value are being restored.
class CalleeSavedAnalysis {
  const FrameAnalysis &FA;
  const BinaryContext &BC;
  BinaryFunction &BF;
  DataflowInfoManager &Info;
  MCPlusBuilder::AllocatorIdTy AllocatorId;

  std::optional<unsigned> SaveTagIndex;
  std::optional<unsigned> RestoreTagIndex;

  /// Compute all stores of callee-saved regs. Those are the ones that stores a
  /// register whose definition is not local.
  void analyzeSaves();

  /// Similar to analyzeSaves, tries to determine all instructions that recover
  /// the original value of the callee-saved register before exiting the
  /// function.
  void analyzeRestores();

  unsigned getSaveTag() {
    if (SaveTagIndex)
      return *SaveTagIndex;
    SaveTagIndex = BC.MIB->getOrCreateAnnotationIndex(getSaveTagName());
    return *SaveTagIndex;
  }

  unsigned getRestoreTag() {
    if (RestoreTagIndex)
      return *RestoreTagIndex;
    RestoreTagIndex = BC.MIB->getOrCreateAnnotationIndex(getRestoreTagName());
    return *RestoreTagIndex;
  }

public:
  BitVector CalleeSaved;
  std::vector<int64_t> OffsetsByReg;
  BitVector HasRestores;
  std::vector<uint64_t> SavingCost;
  std::vector<const FrameIndexEntry *> SaveFIEByReg;
  std::vector<const FrameIndexEntry *> LoadFIEByReg;

  CalleeSavedAnalysis(const FrameAnalysis &FA, BinaryFunction &BF,
                      DataflowInfoManager &Info,
                      MCPlusBuilder::AllocatorIdTy AllocId)
      : FA(FA), BC(BF.getBinaryContext()), BF(BF), Info(Info),
        AllocatorId(AllocId), CalleeSaved(BC.MRI->getNumRegs(), false),
        OffsetsByReg(BC.MRI->getNumRegs(), 0LL),
        HasRestores(BC.MRI->getNumRegs(), false),
        SavingCost(BC.MRI->getNumRegs(), 0ULL),
        SaveFIEByReg(BC.MRI->getNumRegs(), nullptr),
        LoadFIEByReg(BC.MRI->getNumRegs(), nullptr) {}

  ~CalleeSavedAnalysis();

  void compute() {
    analyzeSaves();
    analyzeRestores();
  }

  /// Retrieves the value of the callee-saved register that is saved by this
  /// instruction or 0 if this is not a CSR save instruction.
  uint16_t getSavedReg(const MCInst &Inst) {
    auto Val = BC.MIB->tryGetAnnotationAs<decltype(FrameIndexEntry::RegOrImm)>(
        Inst, getSaveTag());
    if (Val)
      return *Val;
    return 0;
  }

  /// Retrieves the value of the callee-saved register that is restored by this
  /// instruction or 0 if this is not a CSR restore instruction.
  uint16_t getRestoredReg(const MCInst &Inst) {
    auto Val = BC.MIB->tryGetAnnotationAs<decltype(FrameIndexEntry::RegOrImm)>(
        Inst, getRestoreTag());
    if (Val)
      return *Val;
    return 0;
  }

  /// Routines to compute all saves/restores for a Reg (needs to traverse all
  /// instructions).
  std::vector<MCInst *> getSavesByReg(uint16_t Reg);
  std::vector<MCInst *> getRestoresByReg(uint16_t Reg);

  /// Returns the identifying string used to annotate instructions with metadata
  /// for this analysis. These are deleted in the destructor.
  static StringRef getSaveTagName() { return StringRef("CSA-SavedReg"); }

  static StringRef getRestoreTagName() { return StringRef("CSA-RestoredReg"); }
};

/// Identifies in a given binary function all stack regions being used and allow
/// us to edit the layout, removing or inserting new regions. When the layout is
/// modified, all affected stack-accessing instructions are updated.
class StackLayoutModifier {
  const FrameAnalysis &FA;
  const BinaryContext &BC;
  BinaryFunction &BF;
  DataflowInfoManager &Info;
  MCPlusBuilder::AllocatorIdTy AllocatorId;

  // Keep track of stack slots we know how to safely move
  std::map<int64_t, int64_t> AvailableRegions;

  DenseSet<int64_t> CollapsedRegions;
  DenseSet<int64_t> InsertedRegions;

  // A map of chunks of stack memory we don't really know what's happening there
  // and we need to leave it untouched.
  std::map<int64_t, int64_t> BlacklistedRegions;

  // Maps stack slots to the regs that are saved to them
  DenseMap<int64_t, std::set<MCPhysReg>> RegionToRegMap;
  DenseMap<int, std::set<int64_t>> RegToRegionMap;

  // If we can't understand how to move stack slots, IsSimple will be false
  bool IsSimple{true};

  bool IsInitialized{false};

  std::optional<unsigned> TodoTagIndex;
  std::optional<unsigned> SlotTagIndex;
  std::optional<unsigned> OffsetCFIRegTagIndex;

public:
  // Keep a worklist of operations to perform on the function to perform
  // the requested layout modifications via collapseRegion()/insertRegion().
  struct WorklistItem {
    enum ActionType : uint8_t {
      None = 0,
      AdjustLoadStoreOffset,
      AdjustCFI,
    } Action;

    int64_t OffsetUpdate{0};
    WorklistItem() : Action(None) {}
    WorklistItem(ActionType Action) : Action(Action) {}
    WorklistItem(ActionType Action, int OffsetUpdate)
        : Action(Action), OffsetUpdate(OffsetUpdate) {}
  };

private:
  /// Mark the stack region identified by \p Offset and \p Size to be a
  /// no-touch zone, whose accesses cannot be relocated to another region.
  void blacklistRegion(int64_t Offset, int64_t Size);

  /// Check if this region overlaps with blacklisted addresses
  bool isRegionBlacklisted(int64_t Offset, int64_t Size);

  /// Check if the region identified by \p Offset and \p Size has any conflicts
  /// with available regions so far. If it has, blacklist all involved regions
  /// and return true.
  bool blacklistAllInConflictWith(int64_t Offset, int64_t Size);

  /// If \p Point is identified as frame pointer initialization (defining the
  /// value of FP with SP), check for non-standard initialization that precludes
  /// us from changing the stack layout. If positive, update blacklisted
  /// regions.
  void checkFramePointerInitialization(MCInst &Point);

  /// If \p Point is restoring the value with SP with FP plus offset,
  /// add a slottag to this instruction as it needs to be updated when we
  /// change the stack layout.
  void checkStackPointerRestore(MCInst &Point);

  /// Make sense of each stack offsets we can freely change
  void classifyStackAccesses();
  void classifyCFIs();

  /// Used to keep track of modifications to the function that will later be
  /// performed by performChanges();
  void scheduleChange(MCInst &Inst, WorklistItem Item);

  unsigned getTodoTag() {
    if (TodoTagIndex)
      return *TodoTagIndex;
    TodoTagIndex = BC.MIB->getOrCreateAnnotationIndex(getTodoTagName());
    return *TodoTagIndex;
  }

  unsigned getSlotTag() {
    if (SlotTagIndex)
      return *SlotTagIndex;
    SlotTagIndex = BC.MIB->getOrCreateAnnotationIndex(getSlotTagName());
    return *SlotTagIndex;
  }

  unsigned getOffsetCFIRegTag() {
    if (OffsetCFIRegTagIndex)
      return *OffsetCFIRegTagIndex;
    OffsetCFIRegTagIndex =
        BC.MIB->getOrCreateAnnotationIndex(getOffsetCFIRegTagName());
    return *OffsetCFIRegTagIndex;
  }

public:
  StackLayoutModifier(const FrameAnalysis &FA, BinaryFunction &BF,
                      DataflowInfoManager &Info,
                      MCPlusBuilder::AllocatorIdTy AllocId)
      : FA(FA), BC(BF.getBinaryContext()), BF(BF), Info(Info),
        AllocatorId(AllocId) {}

  ~StackLayoutModifier() {
    for (BinaryBasicBlock &BB : BF) {
      for (MCInst &Inst : BB) {
        BC.MIB->removeAnnotation(Inst, getTodoTag());
        BC.MIB->removeAnnotation(Inst, getSlotTag());
        BC.MIB->removeAnnotation(Inst, getOffsetCFIRegTag());
      }
    }
  }

  /// Retrieves the value of the callee-saved register that is restored by this
  /// instruction or 0 if this is not a CSR restore instruction.
  uint16_t getOffsetCFIReg(const MCInst &Inst) {
    auto Val = BC.MIB->tryGetAnnotationAs<uint16_t>(Inst, getOffsetCFIRegTag());
    if (Val)
      return *Val;
    return 0;
  }

  /// Check if it is possible to delete the push instruction \p DeletedPush.
  /// This involves collapsing the region accessed by this push and updating all
  /// other instructions that access affected memory regions. Return true if we
  /// can update this.
  bool canCollapseRegion(int64_t RegionAddr);
  bool canCollapseRegion(MCInst *DeletedPush);

  /// Notify the layout manager that \p DeletedPush was deleted and that it
  /// needs to update other affected stack-accessing instructions.
  bool collapseRegion(MCInst *Alloc, int64_t RegionAddr, int64_t RegionSize);
  bool collapseRegion(MCInst *DeletedPush);

  /// Set the new stack address difference for load/store instructions that
  /// referenced a stack location that was deleted via collapseRegion.
  void setOffsetForCollapsedAccesses(int64_t NewOffset);

  /// Check if it is possible to insert a push instruction at point \p P.
  /// This involves inserting a new region in the stack, possibly affecting
  /// instructions that access the frame. Return true if we can update them all.
  bool canInsertRegion(ProgramPoint P);

  /// Notify the layout manager that a new push instruction has been inserted
  /// at point \p P and that it will need to update relevant instructions.
  bool insertRegion(ProgramPoint P, int64_t RegionSz);

  /// Perform all changes scheduled by collapseRegion()/insertRegion()
  void performChanges();

  /// Perform initial assessment of the function trying to understand its stack
  /// accesses.
  void initialize();

  static StringRef getTodoTagName() { return StringRef("SLM-TodoTag"); }

  static StringRef getSlotTagName() { return StringRef("SLM-SlotTag"); }

  static StringRef getOffsetCFIRegTagName() {
    return StringRef("SLM-OffsetCFIReg");
  }
};

/// Implements a pass to optimize callee-saved register spills. These spills
/// typically happen at function prologue/epilogue. When these are hot basic
/// blocks, this pass will try to move these spills to cold blocks whenever
/// possible.
class ShrinkWrapping {
  const FrameAnalysis &FA;
  const BinaryContext &BC;
  BinaryFunction &BF;
  DataflowInfoManager &Info;
  MCPlusBuilder::AllocatorIdTy AllocatorId;
  StackLayoutModifier SLM;
  /// For each CSR, store a vector of all CFI indexes deleted as a consequence
  /// of moving this Callee-Saved Reg
  DenseMap<unsigned, std::vector<uint32_t>> DeletedPushCFIs;
  DenseMap<unsigned, std::vector<uint32_t>> DeletedPopCFIs;
  BitVector HasDeletedOffsetCFIs;
  SmallPtrSet<const MCCFIInstruction *, 16> UpdatedCFIs;
  std::vector<BitVector> UsesByReg;
  std::vector<int64_t> PushOffsetByReg;
  std::vector<int64_t> PopOffsetByReg;
  std::vector<MCPhysReg> DomOrder;
  CalleeSavedAnalysis CSA;
  std::vector<std::vector<uint64_t>> BestSaveCount;
  std::vector<std::vector<MCInst *>> BestSavePos;

  /// Pass stats
  static std::atomic<std::uint64_t> SpillsMovedRegularMode;
  static std::atomic<std::uint64_t> SpillsMovedPushPopMode;
  static std::atomic<std::uint64_t> SpillsMovedDynamicCount;
  static std::atomic<std::uint64_t> SpillsFailedDynamicCount;
  static std::atomic<std::uint64_t> InstrDynamicCount;
  static std::atomic<std::uint64_t> StoreDynamicCount;

  std::optional<unsigned> AnnotationIndex;

  /// Allow our custom worklist-sensitive analysis
  /// PredictiveStackPointerTracking to access WorklistItem
public:
  struct WorklistItem {
    enum ActionType : uint8_t {
      Erase = 0,
      ChangeToAdjustment,
      InsertLoadOrStore,
      InsertPushOrPop
    } Action;
    FrameIndexEntry FIEToInsert;
    unsigned AffectedReg;
    int Adjustment{0};
    WorklistItem(ActionType Action, unsigned AffectedReg)
        : Action(Action), FIEToInsert(), AffectedReg(AffectedReg) {}
    WorklistItem(ActionType Action, unsigned AffectedReg, int Adjustment)
        : Action(Action), FIEToInsert(), AffectedReg(AffectedReg),
          Adjustment(Adjustment) {}
    WorklistItem(ActionType Action, const FrameIndexEntry &FIE,
                 unsigned AffectedReg)
        : Action(Action), FIEToInsert(FIE), AffectedReg(AffectedReg) {}
  };

  /// Insertion todo items scheduled to happen at the end of BBs. Since we
  /// can't annotate BBs we maintain this bookkeeping here.
  DenseMap<BinaryBasicBlock *, std::vector<WorklistItem>> Todo;

  /// Annotation name used to tag instructions with removal or insertion actions
  static StringRef getAnnotationName() { return StringRef("ShrinkWrap-Todo"); }

  unsigned getAnnotationIndex() {
    if (AnnotationIndex)
      return *AnnotationIndex;
    AnnotationIndex = BC.MIB->getOrCreateAnnotationIndex(getAnnotationName());
    return *AnnotationIndex;
  }

private:
  using BBIterTy = BinaryBasicBlock::iterator;

  /// Calculate all possible uses/defs of these callee-saved regs
  void classifyCSRUses();

  // Ensure we don't work on cases where there are no uses of the callee-saved
  // register. These unnecessary spills should have been removed by previous
  // passes.
  void pruneUnwantedCSRs();

  // Map regs to their possible save possibilities (at start of these BBs)
  void computeSaveLocations();

  /// Look into the best save location found for saving callee-saved reg
  /// \p CSR and evaluates whether we would benefit by moving the spill to this
  /// new save location. Returns true in case it is profitable to perform the
  /// move.
  bool validateBestSavePos(unsigned CSR, MCInst *&BestPosSave,
                           uint64_t &TotalEstimatedWin);

  /// Populate the Todo map with worklistitems to change the function
  template <typename... T> void scheduleChange(ProgramPoint PP, T &&...Item) {
    if (PP.isInst()) {
      auto &WList = BC.MIB->getOrCreateAnnotationAs<std::vector<WorklistItem>>(
          *PP.getInst(), getAnnotationIndex(), AllocatorId);
      WList.emplace_back(std::forward<T>(Item)...);
      return;
    }
    BinaryBasicBlock *BB = PP.getBB();
    // Avoid inserting on BBs with no instructions because we have a dataflow
    // analysis that depends on insertions happening before real instructions
    // (PredictiveStackPointerTracking)
    assert(BB->size() != 0 &&
           "doRestorePlacement() should have handled empty BBs");
    Todo[BB].emplace_back(std::forward<T>(Item)...);
  }

  /// Determine the POP ordering according to which CSR save is the dominator.
  void computeDomOrder();

  /// Check that the best possible location for a spill save (as determined by
  /// computeSaveLocations) is cold enough to be worth moving the save to it.
  /// \p CSR is the callee-saved register number, \p BestPosSave returns the
  /// pointer to the cold location in case the function returns true, while
  /// \p TotalEstimatedWin contains the ins dyn count reduction after moving.
  bool isBestSavePosCold(unsigned CSR, MCInst *&BestPosSave,
                         uint64_t &TotalEstimatedWin);

  /// Auxiliary function used to create basic blocks for critical edges and
  /// update the dominance frontier with these new locations
  void splitFrontierCritEdges(
      BinaryFunction *Func, SmallVector<ProgramPoint, 4> &Frontier,
      const SmallVector<bool, 4> &IsCritEdge,
      const SmallVector<BinaryBasicBlock *, 4> &From,
      const SmallVector<SmallVector<BinaryBasicBlock *, 4>, 4> &To);

  /// After the best save location for a spill has been established in
  /// \p BestPosSave for reg \p CSR, compute adequate locations to restore
  /// the spilled value. This will be at the dominance frontier.
  /// Returns an empty vector if we failed. In case of success, set
  /// \p UsePushPops to true if we can operate in the push/pops mode.
  SmallVector<ProgramPoint, 4> doRestorePlacement(MCInst *BestPosSave,
                                                  unsigned CSR,
                                                  uint64_t TotalEstimatedWin);

  /// Checks whether using push and pops (instead of the longer load-store
  /// counterparts) is correct for reg \p CSR
  bool validatePushPopsMode(unsigned CSR, MCInst *BestPosSave,
                            int64_t SaveOffset);

  /// Adjust restore locations to the correct SP offset if we are using POPs
  /// instead of random-access load instructions.
  SmallVector<ProgramPoint, 4>
  fixPopsPlacements(const SmallVector<ProgramPoint, 4> &RestorePoints,
                    int64_t SaveOffset, unsigned CSR);

  /// When moving spills, mark all old spill locations to be deleted
  void scheduleOldSaveRestoresRemoval(unsigned CSR, bool UsePushPops);
  /// Return true if \p Inst uses reg \p CSR
  bool doesInstUsesCSR(const MCInst &Inst, uint16_t CSR);
  /// When moving spills, mark all new spill locations for insertion
  void
  scheduleSaveRestoreInsertions(unsigned CSR, MCInst *BestPosSave,
                                SmallVector<ProgramPoint, 4> &RestorePoints,
                                bool UsePushPops);

  /// Coordinate the replacement of callee-saved spills from their original
  /// place (at prologue and epilogues) to colder basic blocks as determined
  /// by computeSaveLocations().
  void moveSaveRestores();

  /// Compare multiple basic blocks created by splitting critical edges. If they
  /// have the same contents and successor, fold them into one.
  bool foldIdenticalSplitEdges();

  /// After the spill locations for reg \p CSR has been moved and all affected
  /// CFI has been removed, insert new updated CFI information for these
  /// locations.
  void insertUpdatedCFI(unsigned CSR, int SPValPush, int SPValPop);

  /// In case the function anchors the CFA reg as SP and we inserted pushes/pops
  /// insert def_cfa_offsets at appropriate places (and delete old
  /// def_cfa_offsets)
  void rebuildCFIForSP();

  /// Rebuild all CFI for affected Callee-Saved Registers.
  void rebuildCFI();

  /// Create a load-store instruction (depending on the contents of \p FIE).
  /// If \p CreatePushOrPop is true, create a push/pop instead. Current SP/FP
  /// values, as determined by StackPointerTracking, should be informed via
  /// \p SPVal and \p FPVal in order to emit the correct offset form SP/FP.
  MCInst createStackAccess(int SPVal, int FPVal, const FrameIndexEntry &FIE,
                           bool CreatePushOrPop);

  /// Update the CFI referenced by \p Inst with \p NewOffset, if the CFI has
  /// an offset.
  void updateCFIInstOffset(MCInst &Inst, int64_t NewOffset);

  /// Insert any CFI that should be attached to a register spill save/restore.
  BBIterTy insertCFIsForPushOrPop(BinaryBasicBlock &BB, BBIterTy Pos,
                                  unsigned Reg, bool IsPush, int Sz,
                                  int64_t NewOffset);

  /// Auxiliary function to processInsertionsList, adding a new instruction
  /// before \p InsertionPoint as requested by \p Item. Return an updated
  /// InsertionPoint for other instructions that need to be inserted at the same
  /// original location, since this insertion may have invalidated the previous
  /// location.
  BBIterTy processInsertion(BBIterTy InsertionPoint, BinaryBasicBlock *CurBB,
                            const WorklistItem &Item, int64_t SPVal,
                            int64_t FPVal);

  /// Auxiliary function to processInsertions(), helping perform all the
  /// insertion tasks in the todo list associated with a single insertion point.
  /// Return true if at least one insertion was performed.
  BBIterTy processInsertionsList(BBIterTy InsertionPoint,
                                 BinaryBasicBlock *CurBB,
                                 std::vector<WorklistItem> &TodoList,
                                 int64_t SPVal, int64_t FPVal);

  /// Apply all insertion todo tasks regarding insertion of new stores/loads or
  /// push/pops at annotated points. Return false if the entire function had
  /// no todo tasks annotation and this pass has nothing to do.
  bool processInsertions();

  /// Apply all deletion todo tasks (or tasks to change a push/pop to a memory
  /// access no-op)
  void processDeletions();

public:
  ShrinkWrapping(const FrameAnalysis &FA, BinaryFunction &BF,
                 DataflowInfoManager &Info,
                 MCPlusBuilder::AllocatorIdTy AllocId)
      : FA(FA), BC(BF.getBinaryContext()), BF(BF), Info(Info),
        AllocatorId(AllocId), SLM(FA, BF, Info, AllocId),
        CSA(FA, BF, Info, AllocId) {}

  ~ShrinkWrapping() {
    for (BinaryBasicBlock &BB : BF)
      for (MCInst &Inst : BB)
        BC.MIB->removeAnnotation(Inst, getAnnotationIndex());
  }

  bool perform(bool HotOnly = false);

  static void printStats();
};

} // end namespace bolt
} // end namespace llvm

#endif
