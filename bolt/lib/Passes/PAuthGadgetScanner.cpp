//===- bolt/Passes/PAuthGadgetScanner.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that looks for any AArch64 return instructions
// that may not be protected by PAuth authentication instructions when needed.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/PAuthGadgetScanner.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Passes/DataflowAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Format.h"
#include <memory>

#define DEBUG_TYPE "bolt-pauth-scanner"

namespace llvm {
namespace bolt {

raw_ostream &operator<<(raw_ostream &OS, const MCInstInBBReference &Ref) {
  OS << "MCInstBBRef<";
  if (Ref.BB == nullptr)
    OS << "BB:(null)";
  else
    OS << "BB:" << Ref.BB->getName() << ":" << Ref.BBIndex;
  OS << ">";
  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const MCInstInBFReference &Ref) {
  OS << "MCInstBFRef<";
  if (Ref.BF == nullptr)
    OS << "BF:(null)";
  else
    OS << "BF:" << Ref.BF->getPrintName() << ":" << Ref.getOffset();
  OS << ">";
  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const MCInstReference &Ref) {
  switch (Ref.ParentKind) {
  case MCInstReference::BasicBlockParent:
    OS << Ref.U.BBRef;
    return OS;
  case MCInstReference::FunctionParent:
    OS << Ref.U.BFRef;
    return OS;
  }
  llvm_unreachable("");
}

namespace PAuthGadgetScanner {

[[maybe_unused]] static void traceInst(const BinaryContext &BC, StringRef Label,
                                       const MCInst &MI) {
  dbgs() << "  " << Label << ": ";
  BC.printInstruction(dbgs(), MI);
}

[[maybe_unused]] static void traceReg(const BinaryContext &BC, StringRef Label,
                                      ErrorOr<MCPhysReg> Reg) {
  dbgs() << "    " << Label << ": ";
  if (Reg.getError())
    dbgs() << "(error)";
  else if (*Reg == BC.MIB->getNoRegister())
    dbgs() << "(none)";
  else
    dbgs() << BC.MRI->getName(*Reg);
  dbgs() << "\n";
}

[[maybe_unused]] static void traceRegMask(const BinaryContext &BC,
                                          StringRef Label, BitVector Mask) {
  dbgs() << "    " << Label << ": ";
  RegStatePrinter(BC).print(dbgs(), Mask);
  dbgs() << "\n";
}

// This class represents mapping from a set of arbitrary physical registers to
// consecutive array indexes.
class TrackedRegisters {
  static constexpr uint16_t NoIndex = -1;
  const std::vector<MCPhysReg> Registers;
  std::vector<uint16_t> RegToIndexMapping;

  static size_t getMappingSize(const std::vector<MCPhysReg> &RegsToTrack) {
    if (RegsToTrack.empty())
      return 0;
    return 1 + *llvm::max_element(RegsToTrack);
  }

public:
  TrackedRegisters(const std::vector<MCPhysReg> &RegsToTrack)
      : Registers(RegsToTrack),
        RegToIndexMapping(getMappingSize(RegsToTrack), NoIndex) {
    for (unsigned I = 0; I < RegsToTrack.size(); ++I)
      RegToIndexMapping[RegsToTrack[I]] = I;
  }

  const ArrayRef<MCPhysReg> getRegisters() const { return Registers; }

  size_t getNumTrackedRegisters() const { return Registers.size(); }

  bool empty() const { return Registers.empty(); }

  bool isTracked(MCPhysReg Reg) const {
    bool IsTracked = (unsigned)Reg < RegToIndexMapping.size() &&
                     RegToIndexMapping[Reg] != NoIndex;
    assert(IsTracked == llvm::is_contained(Registers, Reg));
    return IsTracked;
  }

  unsigned getIndex(MCPhysReg Reg) const {
    assert(isTracked(Reg) && "Register is not tracked");
    return RegToIndexMapping[Reg];
  }
};

// The security property that is checked is:
// When a register is used as the address to jump to in a return instruction,
// that register must be safe-to-dereference. It must either
// (a) be safe-to-dereference at function entry and never be changed within this
//     function, i.e. have the same value as when the function started, or
// (b) the last write to the register must be by an authentication instruction.

// This property is checked by using dataflow analysis to keep track of which
// registers have been written (def-ed), since last authenticated. For pac-ret,
// any return instruction using a register which is not safe-to-dereference is
// a gadget to be reported. For PAuthABI, probably at least any indirect control
// flow using such a register should be reported.

// Furthermore, when producing a diagnostic for a found non-pac-ret protected
// return, the analysis also lists the last instructions that wrote to the
// register used in the return instruction.
// The total set of registers used in return instructions in a given function is
// small. It almost always is just `X30`.
// In order to reduce the memory consumption of storing this additional state
// during the dataflow analysis, this is computed by running the dataflow
// analysis twice:
// 1. In the first run, the dataflow analysis only keeps track of the security
//    property: i.e. which registers have been overwritten since the last
//    time they've been authenticated.
// 2. If the first run finds any return instructions using a register last
//    written by a non-authenticating instruction, the dataflow analysis will
//    be run a second time. The first run will return which registers are used
//    in the gadgets to be reported. This information is used in the second run
//    to also track which instructions last wrote to those registers.

/// A state representing which registers are safe to use by an instruction
/// at a given program point.
///
/// To simplify reasoning, let's stick with the following approach:
/// * when state is updated by the data-flow analysis, the sub-, super- and
///   overlapping registers are marked as needed
/// * when the particular instruction is checked if it represents a gadget,
///   the specific bit of BitVector should be usable to answer this.
///
/// For example, on AArch64:
/// * An AUTIZA X0 instruction marks both X0 and W0 (as well as W0_HI) as
///   safe-to-dereference. It does not change the state of X0_X1, for example,
///   as super-registers partially retain their old, unsafe values.
/// * LDR X1, [X0] marks as unsafe both X1 itself and anything it overlaps
///   with: W1, W1_HI, X0_X1 and so on.
/// * RET (which is implicitly RET X30) is a protected return if and only if
///   X30 is safe-to-dereference - the state computed for sub- and
///   super-registers is not inspected.
struct State {
  /// A BitVector containing the registers that are either safe at function
  /// entry and were not clobbered yet, or those not clobbered since being
  /// authenticated.
  BitVector SafeToDerefRegs;
  /// A vector of sets, only used in the second data flow run.
  /// Each element in the vector represents one of the registers for which we
  /// track the set of last instructions that wrote to this register. For
  /// pac-ret analysis, the expectation is that almost all return instructions
  /// only use register `X30`, and therefore, this vector will probably have
  /// length 1 in the second run.
  std::vector<SmallPtrSet<const MCInst *, 4>> LastInstWritingReg;

  /// Construct an empty state.
  State() {}

  State(unsigned NumRegs, unsigned NumRegsToTrack)
      : SafeToDerefRegs(NumRegs), LastInstWritingReg(NumRegsToTrack) {}

  State &merge(const State &StateIn) {
    if (StateIn.empty())
      return *this;
    if (empty())
      return (*this = StateIn);

    SafeToDerefRegs &= StateIn.SafeToDerefRegs;
    for (unsigned I = 0; I < LastInstWritingReg.size(); ++I)
      for (const MCInst *J : StateIn.LastInstWritingReg[I])
        LastInstWritingReg[I].insert(J);
    return *this;
  }

  /// Returns true if this object does not store state of any registers -
  /// neither safe, nor unsafe ones.
  bool empty() const { return SafeToDerefRegs.empty(); }

  bool operator==(const State &RHS) const {
    return SafeToDerefRegs == RHS.SafeToDerefRegs &&
           LastInstWritingReg == RHS.LastInstWritingReg;
  }
  bool operator!=(const State &RHS) const { return !((*this) == RHS); }
};

static void printLastInsts(
    raw_ostream &OS,
    const std::vector<SmallPtrSet<const MCInst *, 4>> &LastInstWritingReg) {
  OS << "Insts: ";
  for (unsigned I = 0; I < LastInstWritingReg.size(); ++I) {
    auto &Set = LastInstWritingReg[I];
    OS << "[" << I << "](";
    for (const MCInst *MCInstP : Set)
      OS << MCInstP << " ";
    OS << ")";
  }
}

raw_ostream &operator<<(raw_ostream &OS, const State &S) {
  OS << "pacret-state<";
  if (S.empty()) {
    OS << "empty";
  } else {
    OS << "SafeToDerefRegs: " << S.SafeToDerefRegs << ", ";
    printLastInsts(OS, S.LastInstWritingReg);
  }
  OS << ">";
  return OS;
}

class PacStatePrinter {
public:
  void print(raw_ostream &OS, const State &State) const;
  explicit PacStatePrinter(const BinaryContext &BC) : BC(BC) {}

private:
  const BinaryContext &BC;
};

void PacStatePrinter::print(raw_ostream &OS, const State &S) const {
  RegStatePrinter RegStatePrinter(BC);
  OS << "pacret-state<";
  if (S.empty()) {
    assert(S.SafeToDerefRegs.empty());
    assert(S.LastInstWritingReg.empty());
    OS << "empty";
  } else {
    OS << "SafeToDerefRegs: ";
    RegStatePrinter.print(OS, S.SafeToDerefRegs);
    OS << ", ";
    printLastInsts(OS, S.LastInstWritingReg);
  }
  OS << ">";
}

class PacRetAnalysis
    : public DataflowAnalysis<PacRetAnalysis, State, /*Backward=*/false,
                              PacStatePrinter> {
  using Parent =
      DataflowAnalysis<PacRetAnalysis, State, false, PacStatePrinter>;
  friend Parent;

public:
  PacRetAnalysis(BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocId,
                 const std::vector<MCPhysReg> &RegsToTrackInstsFor)
      : Parent(BF, AllocId), NumRegs(BF.getBinaryContext().MRI->getNumRegs()),
        RegsToTrackInstsFor(RegsToTrackInstsFor) {}
  virtual ~PacRetAnalysis() {}

protected:
  const unsigned NumRegs;
  /// RegToTrackInstsFor is the set of registers for which the dataflow analysis
  /// must compute which the last set of instructions writing to it are.
  const TrackedRegisters RegsToTrackInstsFor;

  SmallPtrSet<const MCInst *, 4> &lastWritingInsts(State &S,
                                                   MCPhysReg Reg) const {
    unsigned Index = RegsToTrackInstsFor.getIndex(Reg);
    return S.LastInstWritingReg[Index];
  }
  const SmallPtrSet<const MCInst *, 4> &lastWritingInsts(const State &S,
                                                         MCPhysReg Reg) const {
    unsigned Index = RegsToTrackInstsFor.getIndex(Reg);
    return S.LastInstWritingReg[Index];
  }

  void preflight() {}

  State createEntryState() {
    State S(NumRegs, RegsToTrackInstsFor.getNumTrackedRegisters());
    for (MCPhysReg Reg : BC.MIB->getTrustedLiveInRegs())
      S.SafeToDerefRegs |= BC.MIB->getAliases(Reg, /*OnlySmaller=*/true);
    return S;
  }

  State getStartingStateAtBB(const BinaryBasicBlock &BB) {
    if (BB.isEntryPoint())
      return createEntryState();

    return State();
  }

  State getStartingStateAtPoint(const MCInst &Point) { return State(); }

  void doConfluence(State &StateOut, const State &StateIn) {
    PacStatePrinter P(BC);
    LLVM_DEBUG({
      dbgs() << " PacRetAnalysis::Confluence(\n";
      dbgs() << "   State 1: ";
      P.print(dbgs(), StateOut);
      dbgs() << "\n";
      dbgs() << "   State 2: ";
      P.print(dbgs(), StateIn);
      dbgs() << ")\n";
    });

    StateOut.merge(StateIn);

    LLVM_DEBUG({
      dbgs() << "   merged state: ";
      P.print(dbgs(), StateOut);
      dbgs() << "\n";
    });
  }

  BitVector getClobberedRegs(const MCInst &Point) const {
    BitVector Clobbered(NumRegs, false);
    // Assume a call can clobber all registers, including callee-saved
    // registers. There's a good chance that callee-saved registers will be
    // saved on the stack at some point during execution of the callee.
    // Therefore they should also be considered as potentially modified by an
    // attacker/written to.
    // Also, not all functions may respect the AAPCS ABI rules about
    // caller/callee-saved registers.
    if (BC.MIB->isCall(Point))
      Clobbered.set();
    else
      BC.MIB->getClobberedRegs(Point, Clobbered);
    return Clobbered;
  }

  // Returns all registers that can be treated as if they are written by an
  // authentication instruction.
  SmallVector<MCPhysReg> getRegsMadeSafeToDeref(const MCInst &Point,
                                                const State &Cur) const {
    SmallVector<MCPhysReg> Regs;
    const MCPhysReg NoReg = BC.MIB->getNoRegister();

    // A signed pointer can be authenticated, or
    ErrorOr<MCPhysReg> AutReg = BC.MIB->getAuthenticatedReg(Point);
    if (AutReg && *AutReg != NoReg)
      Regs.push_back(*AutReg);

    // ... a safe address can be materialized, or
    MCPhysReg NewAddrReg = BC.MIB->getMaterializedAddressRegForPtrAuth(Point);
    if (NewAddrReg != NoReg)
      Regs.push_back(NewAddrReg);

    // ... an address can be updated in a safe manner, producing the result
    // which is as trusted as the input address.
    if (auto DstAndSrc = BC.MIB->analyzeAddressArithmeticsForPtrAuth(Point)) {
      if (Cur.SafeToDerefRegs[DstAndSrc->second])
        Regs.push_back(DstAndSrc->first);
    }

    return Regs;
  }

  State computeNext(const MCInst &Point, const State &Cur) {
    PacStatePrinter P(BC);
    LLVM_DEBUG({
      dbgs() << " PacRetAnalysis::ComputeNext(";
      BC.InstPrinter->printInst(&const_cast<MCInst &>(Point), 0, "", *BC.STI,
                                dbgs());
      dbgs() << ", ";
      P.print(dbgs(), Cur);
      dbgs() << ")\n";
    });

    // If this instruction is reachable, a non-empty state will be propagated
    // to it from the entry basic block sooner or later. Until then, it is both
    // more efficient and easier to reason about to skip computeNext().
    if (Cur.empty()) {
      LLVM_DEBUG(
          { dbgs() << "Skipping computeNext(Point, Cur) as Cur is empty.\n"; });
      return State();
    }

    // First, compute various properties of the instruction, taking the state
    // before its execution into account, if necessary.

    BitVector Clobbered = getClobberedRegs(Point);
    SmallVector<MCPhysReg> NewSafeToDerefRegs =
        getRegsMadeSafeToDeref(Point, Cur);

    // Then, compute the state after this instruction is executed.
    State Next = Cur;

    Next.SafeToDerefRegs.reset(Clobbered);
    // Keep track of this instruction if it writes to any of the registers we
    // need to track that for:
    for (MCPhysReg Reg : RegsToTrackInstsFor.getRegisters())
      if (Clobbered[Reg])
        lastWritingInsts(Next, Reg) = {&Point};

    // After accounting for clobbered registers in general, override the state
    // according to authentication and other *special cases* of clobbering.

    // The sub-registers are also safe-to-dereference now, but not their
    // super-registers (as they retain untrusted register units).
    BitVector NewSafeSubregs(NumRegs);
    for (MCPhysReg SafeReg : NewSafeToDerefRegs)
      NewSafeSubregs |= BC.MIB->getAliases(SafeReg, /*OnlySmaller=*/true);
    for (MCPhysReg Reg : NewSafeSubregs.set_bits()) {
      Next.SafeToDerefRegs.set(Reg);
      if (RegsToTrackInstsFor.isTracked(Reg))
        lastWritingInsts(Next, Reg).clear();
    }

    LLVM_DEBUG({
      dbgs() << "  .. result: (";
      P.print(dbgs(), Next);
      dbgs() << ")\n";
    });

    return Next;
  }

  StringRef getAnnotationName() const { return StringRef("PacRetAnalysis"); }

public:
  std::vector<MCInstReference>
  getLastClobberingInsts(const MCInst &Inst, BinaryFunction &BF,
                         const ArrayRef<MCPhysReg> UsedDirtyRegs) const {
    if (RegsToTrackInstsFor.empty())
      return {};
    auto MaybeState = getStateBefore(Inst);
    if (!MaybeState)
      llvm_unreachable("Expected State to be present");
    const State &S = *MaybeState;
    // Due to aliasing registers, multiple registers may have been tracked.
    std::set<const MCInst *> LastWritingInsts;
    for (MCPhysReg TrackedReg : UsedDirtyRegs) {
      for (const MCInst *Inst : lastWritingInsts(S, TrackedReg))
        LastWritingInsts.insert(Inst);
    }
    std::vector<MCInstReference> Result;
    for (const MCInst *Inst : LastWritingInsts) {
      MCInstInBBReference Ref = MCInstInBBReference::get(Inst, BF);
      assert(Ref.BB != nullptr && "Expected Inst to be found");
      Result.push_back(MCInstReference(Ref));
    }
    return Result;
  }
};

static std::shared_ptr<Report>
shouldReportReturnGadget(const BinaryContext &BC, const MCInstReference &Inst,
                         const State &S) {
  static const GadgetKind RetKind("non-protected ret found");
  if (!BC.MIB->isReturn(Inst))
    return nullptr;

  ErrorOr<MCPhysReg> MaybeRetReg = BC.MIB->getRegUsedAsRetDest(Inst);
  if (MaybeRetReg.getError()) {
    return std::make_shared<GenericReport>(
        Inst, "Warning: pac-ret analysis could not analyze this return "
              "instruction");
  }
  MCPhysReg RetReg = *MaybeRetReg;
  LLVM_DEBUG({
    traceInst(BC, "Found RET inst", Inst);
    traceReg(BC, "RetReg", RetReg);
    traceReg(BC, "Authenticated reg", BC.MIB->getAuthenticatedReg(Inst));
  });
  if (BC.MIB->isAuthenticationOfReg(Inst, RetReg))
    return nullptr;
  LLVM_DEBUG({ traceRegMask(BC, "SafeToDerefRegs", S.SafeToDerefRegs); });
  if (S.SafeToDerefRegs[RetReg])
    return nullptr;

  return std::make_shared<GadgetReport>(RetKind, Inst, RetReg);
}

static std::shared_ptr<Report>
shouldReportCallGadget(const BinaryContext &BC, const MCInstReference &Inst,
                       const State &S) {
  static const GadgetKind CallKind("non-protected call found");
  if (!BC.MIB->isCall(Inst) && !BC.MIB->isBranch(Inst))
    return nullptr;

  bool IsAuthenticated = false;
  MCPhysReg DestReg = BC.MIB->getRegUsedAsCallDest(Inst, IsAuthenticated);
  if (IsAuthenticated || DestReg == BC.MIB->getNoRegister())
    return nullptr;

  LLVM_DEBUG({
    traceInst(BC, "Found call inst", Inst);
    traceReg(BC, "Call destination reg", DestReg);
    traceRegMask(BC, "SafeToDerefRegs", S.SafeToDerefRegs);
  });
  if (S.SafeToDerefRegs[DestReg])
    return nullptr;

  return std::make_shared<GadgetReport>(CallKind, Inst, DestReg);
}

FunctionAnalysisResult
Analysis::findGadgets(BinaryFunction &BF,
                      MCPlusBuilder::AllocatorIdTy AllocatorId) {
  FunctionAnalysisResult Result;

  PacRetAnalysis PRA(BF, AllocatorId, {});
  PRA.run();
  LLVM_DEBUG({
    dbgs() << " After PacRetAnalysis:\n";
    BF.dump();
  });

  BinaryContext &BC = BF.getBinaryContext();
  for (BinaryBasicBlock &BB : BF) {
    for (int64_t I = 0, E = BB.size(); I < E; ++I) {
      MCInstReference Inst(&BB, I);
      const State &S = *PRA.getStateBefore(Inst);

      // If non-empty state was never propagated from the entry basic block
      // to Inst, assume it to be unreachable and report a warning.
      if (S.empty()) {
        Result.Diagnostics.push_back(std::make_shared<GenericReport>(
            Inst, "Warning: unreachable instruction found"));
        continue;
      }

      if (auto Report = shouldReportReturnGadget(BC, Inst, S))
        Result.Diagnostics.push_back(Report);

      if (PacRetGadgetsOnly)
        continue;

      if (auto Report = shouldReportCallGadget(BC, Inst, S))
        Result.Diagnostics.push_back(Report);
    }
  }
  return Result;
}

void Analysis::computeDetailedInfo(BinaryFunction &BF,
                                   MCPlusBuilder::AllocatorIdTy AllocatorId,
                                   FunctionAnalysisResult &Result) {
  BinaryContext &BC = BF.getBinaryContext();

  // Collect the affected registers across all gadgets found in this function.
  SmallSet<MCPhysReg, 4> RegsToTrack;
  for (auto Report : Result.Diagnostics)
    RegsToTrack.insert_range(Report->getAffectedRegisters());
  std::vector<MCPhysReg> RegsToTrackVec(RegsToTrack.begin(), RegsToTrack.end());

  // Re-compute the analysis with register tracking.
  PacRetAnalysis PRWIA(BF, AllocatorId, RegsToTrackVec);
  PRWIA.run();
  LLVM_DEBUG({
    dbgs() << " After detailed PacRetAnalysis:\n";
    BF.dump();
  });

  // Augment gadget reports.
  for (auto Report : Result.Diagnostics) {
    LLVM_DEBUG(
        { traceInst(BC, "Attaching clobbering info to", Report->Location); });
    (void)BC;
    Report->setOverwritingInstrs(PRWIA.getLastClobberingInsts(
        Report->Location, BF, Report->getAffectedRegisters()));
  }
}

void Analysis::runOnFunction(BinaryFunction &BF,
                             MCPlusBuilder::AllocatorIdTy AllocatorId) {
  LLVM_DEBUG({
    dbgs() << "Analyzing in function " << BF.getPrintName() << ", AllocatorId "
           << AllocatorId << "\n";
    BF.dump();
  });

  if (!BF.hasCFG())
    return;

  FunctionAnalysisResult FAR = findGadgets(BF, AllocatorId);
  if (FAR.Diagnostics.empty())
    return;

  // Redo the analysis, but now also track which instructions last wrote
  // to any of the registers in RetRegsWithGadgets, so that better
  // diagnostics can be produced.

  computeDetailedInfo(BF, AllocatorId, FAR);

  // `runOnFunction` is typically getting called from multiple threads in
  // parallel. Therefore, use a lock to avoid data races when storing the
  // result of the analysis in the `AnalysisResults` map.
  {
    std::lock_guard<std::mutex> Lock(AnalysisResultsMutex);
    AnalysisResults[&BF] = FAR;
  }
}

static void printBB(const BinaryContext &BC, const BinaryBasicBlock *BB,
                    size_t StartIndex = 0, size_t EndIndex = -1) {
  if (EndIndex == (size_t)-1)
    EndIndex = BB->size() - 1;
  const BinaryFunction *BF = BB->getFunction();
  for (unsigned I = StartIndex; I <= EndIndex; ++I) {
    // FIXME: this assumes all instructions are 4 bytes in size. This is true
    // for AArch64, but it might be good to extract this function so it can be
    // used elsewhere and for other targets too.
    uint64_t Address = BB->getOffset() + BF->getAddress() + 4 * I;
    const MCInst &Inst = BB->getInstructionAtIndex(I);
    if (BC.MIB->isCFI(Inst))
      continue;
    BC.printInstruction(outs(), Inst, Address, BF);
  }
}

static void reportFoundGadgetInSingleBBSingleOverwInst(
    raw_ostream &OS, const BinaryContext &BC, const MCInstReference OverwInst,
    const MCInstReference Location) {
  BinaryBasicBlock *BB = Location.getBasicBlock();
  assert(OverwInst.ParentKind == MCInstReference::BasicBlockParent);
  assert(Location.ParentKind == MCInstReference::BasicBlockParent);
  MCInstInBBReference OverwInstBB = OverwInst.U.BBRef;
  if (BB == OverwInstBB.BB) {
    // overwriting inst and ret instruction are in the same basic block.
    assert(OverwInstBB.BBIndex < Location.U.BBRef.BBIndex);
    OS << "  This happens in the following basic block:\n";
    printBB(BC, BB);
  }
}

void Report::printBasicInfo(raw_ostream &OS, const BinaryContext &BC,
                            StringRef IssueKind) const {
  BinaryFunction *BF = Location.getFunction();
  BinaryBasicBlock *BB = Location.getBasicBlock();

  OS << "\nGS-PAUTH: " << IssueKind;
  OS << " in function " << BF->getPrintName();
  if (BB)
    OS << ", basic block " << BB->getName();
  OS << ", at address " << llvm::format("%x", Location.getAddress()) << "\n";
  OS << "  The instruction is ";
  BC.printInstruction(OS, Location, Location.getAddress(), BF);
}

void GadgetReport::generateReport(raw_ostream &OS,
                                  const BinaryContext &BC) const {
  printBasicInfo(OS, BC, Kind.getDescription());

  BinaryFunction *BF = Location.getFunction();
  OS << "  The " << OverwritingInstrs.size()
     << " instructions that write to the affected registers after any "
        "authentication are:\n";
  // Sort by address to ensure output is deterministic.
  SmallVector<MCInstReference> OI = OverwritingInstrs;
  llvm::sort(OI, [](const MCInstReference &A, const MCInstReference &B) {
    return A.getAddress() < B.getAddress();
  });
  for (unsigned I = 0; I < OI.size(); ++I) {
    MCInstReference InstRef = OI[I];
    OS << "  " << (I + 1) << ". ";
    BC.printInstruction(OS, InstRef, InstRef.getAddress(), BF);
  };
  if (OverwritingInstrs.size() == 1) {
    const MCInstReference OverwInst = OverwritingInstrs[0];
    assert(OverwInst.ParentKind == MCInstReference::BasicBlockParent);
    reportFoundGadgetInSingleBBSingleOverwInst(OS, BC, OverwInst, Location);
  }
}

void GenericReport::generateReport(raw_ostream &OS,
                                   const BinaryContext &BC) const {
  printBasicInfo(OS, BC, Text);
}

Error Analysis::runOnFunctions(BinaryContext &BC) {
  ParallelUtilities::WorkFuncWithAllocTy WorkFun =
      [&](BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocatorId) {
        runOnFunction(BF, AllocatorId);
      };

  ParallelUtilities::PredicateTy SkipFunc = [&](const BinaryFunction &BF) {
    return false;
  };

  ParallelUtilities::runOnEachFunctionWithUniqueAllocId(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_LINEAR, WorkFun,
      SkipFunc, "PAuthGadgetScanner");

  for (BinaryFunction *BF : BC.getAllBinaryFunctions())
    if (AnalysisResults.count(BF) > 0) {
      for (const std::shared_ptr<Report> &R : AnalysisResults[BF].Diagnostics)
        R->generateReport(outs(), BC);
    }
  return Error::success();
}

} // namespace PAuthGadgetScanner
} // namespace bolt
} // namespace llvm
