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
namespace PAuthGadgetScanner {

[[maybe_unused]] static void traceInst(const BinaryContext &BC, StringRef Label,
                                       const MCInst &MI) {
  dbgs() << "  " << Label << ": ";
  BC.printInstruction(dbgs(), MI);
}

[[maybe_unused]] static void traceReg(const BinaryContext &BC, StringRef Label,
                                      MCPhysReg Reg) {
  dbgs() << "    " << Label << ": ";
  if (Reg == BC.MIB->getNoRegister())
    dbgs() << "(none)";
  else
    dbgs() << BC.MRI->getName(Reg);
  dbgs() << "\n";
}

[[maybe_unused]] static void traceRegMask(const BinaryContext &BC,
                                          StringRef Label, BitVector Mask) {
  dbgs() << "    " << Label << ": ";
  RegStatePrinter(BC).print(dbgs(), Mask);
  dbgs() << "\n";
}

// Iterates over BinaryFunction's instructions like a range-based for loop:
//
// iterateOverInstrs(BF, [&](MCInstReference Inst) {
//   // loop body
// });
template <typename T> static void iterateOverInstrs(BinaryFunction &BF, T Fn) {
  if (BF.hasCFG()) {
    for (BinaryBasicBlock &BB : BF)
      for (int64_t I = 0, E = BB.size(); I < E; ++I)
        Fn(MCInstReference(&BB, I));
  } else {
    for (auto I = BF.instrs().begin(), E = BF.instrs().end(); I != E; ++I)
      Fn(MCInstReference(&BF, I));
  }
}

// This class represents mapping from a set of arbitrary physical registers to
// consecutive array indexes.
class TrackedRegisters {
  static constexpr uint16_t NoIndex = -1;
  const std::vector<MCPhysReg> Registers;
  std::vector<uint16_t> RegToIndexMapping;

  static size_t getMappingSize(ArrayRef<MCPhysReg> RegsToTrack) {
    if (RegsToTrack.empty())
      return 0;
    return 1 + *llvm::max_element(RegsToTrack);
  }

public:
  TrackedRegisters(ArrayRef<MCPhysReg> RegsToTrack)
      : Registers(RegsToTrack),
        RegToIndexMapping(getMappingSize(RegsToTrack), NoIndex) {
    for (unsigned I = 0; I < RegsToTrack.size(); ++I)
      RegToIndexMapping[RegsToTrack[I]] = I;
  }

  ArrayRef<MCPhysReg> getRegisters() const { return Registers; }

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

typedef SmallPtrSet<const MCInst *, 4> SetOfRelatedInsts;

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
struct SrcState {
  /// A BitVector containing the registers that are either authenticated
  /// (assuming failed authentication is permitted to produce an invalid
  /// address, provided it generates an error on memory access) or whose
  /// value is known not to be attacker-controlled under Pointer Authentication
  /// threat model. The registers in this set are either
  /// * not clobbered since being authenticated, or
  /// * trusted at function entry and were not clobbered yet, or
  /// * contain a safely materialized address.
  BitVector SafeToDerefRegs;
  /// A BitVector containing the registers that are either authenticated
  /// *successfully* or whose value is known not to be attacker-controlled
  /// under Pointer Authentication threat model.
  /// The registers in this set are either
  /// * authenticated and then checked to be authenticated successfully
  ///   (and not clobbered since then), or
  /// * trusted at function entry and were not clobbered yet, or
  /// * contain a safely materialized address.
  BitVector TrustedRegs;
  /// A vector of sets, only used in the second data flow run.
  /// Each element in the vector represents one of the registers for which we
  /// track the set of last instructions that wrote to this register. For
  /// pac-ret analysis, the expectation is that almost all return instructions
  /// only use register `X30`, and therefore, this vector will probably have
  /// length 1 in the second run.
  std::vector<SetOfRelatedInsts> LastInstWritingReg;

  /// Construct an empty state.
  SrcState() {}

  SrcState(unsigned NumRegs, unsigned NumRegsToTrack)
      : SafeToDerefRegs(NumRegs), TrustedRegs(NumRegs),
        LastInstWritingReg(NumRegsToTrack) {}

  SrcState &merge(const SrcState &StateIn) {
    if (StateIn.empty())
      return *this;
    if (empty())
      return (*this = StateIn);

    SafeToDerefRegs &= StateIn.SafeToDerefRegs;
    TrustedRegs &= StateIn.TrustedRegs;
    for (unsigned I = 0; I < LastInstWritingReg.size(); ++I)
      for (const MCInst *J : StateIn.LastInstWritingReg[I])
        LastInstWritingReg[I].insert(J);
    return *this;
  }

  /// Returns true if this object does not store state of any registers -
  /// neither safe, nor unsafe ones.
  bool empty() const { return SafeToDerefRegs.empty(); }

  bool operator==(const SrcState &RHS) const {
    return SafeToDerefRegs == RHS.SafeToDerefRegs &&
           TrustedRegs == RHS.TrustedRegs &&
           LastInstWritingReg == RHS.LastInstWritingReg;
  }
  bool operator!=(const SrcState &RHS) const { return !((*this) == RHS); }
};

static void printInstsShort(raw_ostream &OS,
                            ArrayRef<SetOfRelatedInsts> Insts) {
  OS << "Insts: ";
  for (unsigned I = 0; I < Insts.size(); ++I) {
    auto &Set = Insts[I];
    OS << "[" << I << "](";
    for (const MCInst *MCInstP : Set)
      OS << MCInstP << " ";
    OS << ")";
  }
}

static raw_ostream &operator<<(raw_ostream &OS, const SrcState &S) {
  OS << "src-state<";
  if (S.empty()) {
    OS << "empty";
  } else {
    OS << "SafeToDerefRegs: " << S.SafeToDerefRegs << ", ";
    OS << "TrustedRegs: " << S.TrustedRegs << ", ";
    printInstsShort(OS, S.LastInstWritingReg);
  }
  OS << ">";
  return OS;
}

class SrcStatePrinter {
public:
  void print(raw_ostream &OS, const SrcState &State) const;
  explicit SrcStatePrinter(const BinaryContext &BC) : BC(BC) {}

private:
  const BinaryContext &BC;
};

void SrcStatePrinter::print(raw_ostream &OS, const SrcState &S) const {
  RegStatePrinter RegStatePrinter(BC);
  OS << "src-state<";
  if (S.empty()) {
    assert(S.SafeToDerefRegs.empty());
    assert(S.TrustedRegs.empty());
    assert(S.LastInstWritingReg.empty());
    OS << "empty";
  } else {
    OS << "SafeToDerefRegs: ";
    RegStatePrinter.print(OS, S.SafeToDerefRegs);
    OS << ", TrustedRegs: ";
    RegStatePrinter.print(OS, S.TrustedRegs);
    OS << ", ";
    printInstsShort(OS, S.LastInstWritingReg);
  }
  OS << ">";
}

/// Computes which registers are safe to be used by control flow and signing
/// instructions.
///
/// This is the base class for two implementations: a dataflow-based analysis
/// which is intended to be used for most functions and a simplified CFG-unaware
/// version for functions without reconstructed CFG.
class SrcSafetyAnalysis {
public:
  SrcSafetyAnalysis(BinaryFunction &BF, ArrayRef<MCPhysReg> RegsToTrackInstsFor)
      : BC(BF.getBinaryContext()), NumRegs(BC.MRI->getNumRegs()),
        RegsToTrackInstsFor(RegsToTrackInstsFor) {}

  virtual ~SrcSafetyAnalysis() {}

  static std::shared_ptr<SrcSafetyAnalysis>
  create(BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocId,
         ArrayRef<MCPhysReg> RegsToTrackInstsFor);

  virtual void run() = 0;
  virtual const SrcState &getStateBefore(const MCInst &Inst) const = 0;

protected:
  BinaryContext &BC;
  const unsigned NumRegs;
  /// RegToTrackInstsFor is the set of registers for which the dataflow analysis
  /// must compute which the last set of instructions writing to it are.
  const TrackedRegisters RegsToTrackInstsFor;
  /// Stores information about the detected instruction sequences emitted to
  /// check an authenticated pointer. Specifically, if such sequence is detected
  /// in a basic block, it maps the last instruction of that basic block to
  /// (CheckedRegister, FirstInstOfTheSequence) pair, see the description of
  /// MCPlusBuilder::getAuthCheckedReg(BB) method.
  ///
  /// As the detection of such sequences requires iterating over the adjacent
  /// instructions, it should be done before calling computeNext(), which
  /// operates on separate instructions.
  DenseMap<const MCInst *, std::pair<MCPhysReg, const MCInst *>>
      CheckerSequenceInfo;

  SetOfRelatedInsts &lastWritingInsts(SrcState &S, MCPhysReg Reg) const {
    unsigned Index = RegsToTrackInstsFor.getIndex(Reg);
    return S.LastInstWritingReg[Index];
  }
  const SetOfRelatedInsts &lastWritingInsts(const SrcState &S,
                                            MCPhysReg Reg) const {
    unsigned Index = RegsToTrackInstsFor.getIndex(Reg);
    return S.LastInstWritingReg[Index];
  }

  SrcState createEntryState() {
    SrcState S(NumRegs, RegsToTrackInstsFor.getNumTrackedRegisters());
    for (MCPhysReg Reg : BC.MIB->getTrustedLiveInRegs())
      S.TrustedRegs |= BC.MIB->getAliases(Reg, /*OnlySmaller=*/true);
    S.SafeToDerefRegs = S.TrustedRegs;
    return S;
  }

  /// Computes a reasonably pessimistic estimation of the register state when
  /// the previous instruction is not known for sure. Takes the set of registers
  /// which are trusted at function entry and removes all registers that can be
  /// clobbered inside this function.
  SrcState computePessimisticState(BinaryFunction &BF) {
    BitVector ClobberedRegs(NumRegs);
    iterateOverInstrs(BF, [&](MCInstReference Inst) {
      BC.MIB->getClobberedRegs(Inst, ClobberedRegs);

      // If this is a call instruction, no register is safe anymore, unless
      // it is a tail call. Ignore tail calls for the purpose of estimating the
      // worst-case scenario, assuming no instructions are executed in the
      // caller after this point anyway.
      if (BC.MIB->isCall(Inst) && !BC.MIB->isTailCall(Inst))
        ClobberedRegs.set();
    });

    SrcState S = createEntryState();
    S.SafeToDerefRegs.reset(ClobberedRegs);
    S.TrustedRegs.reset(ClobberedRegs);
    return S;
  }

  BitVector getClobberedRegs(const MCInst &Point) const {
    BitVector Clobbered(NumRegs);
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
                                                const SrcState &Cur) const {
    SmallVector<MCPhysReg> Regs;

    // A signed pointer can be authenticated, or
    bool Dummy = false;
    if (auto AutReg = BC.MIB->getWrittenAuthenticatedReg(Point, Dummy))
      Regs.push_back(*AutReg);

    // ... a safe address can be materialized, or
    if (auto NewAddrReg = BC.MIB->getMaterializedAddressRegForPtrAuth(Point))
      Regs.push_back(*NewAddrReg);

    // ... an address can be updated in a safe manner, producing the result
    // which is as trusted as the input address.
    if (auto DstAndSrc = BC.MIB->analyzeAddressArithmeticsForPtrAuth(Point)) {
      if (Cur.SafeToDerefRegs[DstAndSrc->second])
        Regs.push_back(DstAndSrc->first);
    }

    return Regs;
  }

  // Returns all registers made trusted by this instruction.
  SmallVector<MCPhysReg> getRegsMadeTrusted(const MCInst &Point,
                                            const SrcState &Cur) const {
    SmallVector<MCPhysReg> Regs;

    // An authenticated pointer can be checked, or
    std::optional<MCPhysReg> CheckedReg =
        BC.MIB->getAuthCheckedReg(Point, /*MayOverwrite=*/false);
    if (CheckedReg && Cur.SafeToDerefRegs[*CheckedReg])
      Regs.push_back(*CheckedReg);

    // ... a pointer can be authenticated by an instruction that always checks
    // the pointer, or
    bool IsChecked = false;
    std::optional<MCPhysReg> AutReg =
        BC.MIB->getWrittenAuthenticatedReg(Point, IsChecked);
    if (AutReg && IsChecked)
      Regs.push_back(*AutReg);

    if (CheckerSequenceInfo.contains(&Point)) {
      MCPhysReg CheckedReg;
      const MCInst *FirstCheckerInst;
      std::tie(CheckedReg, FirstCheckerInst) = CheckerSequenceInfo.at(&Point);

      // FirstCheckerInst should belong to the same basic block (see the
      // assertion in DataflowSrcSafetyAnalysis::run()), meaning it was
      // deterministically processed a few steps before this instruction.
      const SrcState &StateBeforeChecker = getStateBefore(*FirstCheckerInst);
      if (StateBeforeChecker.SafeToDerefRegs[CheckedReg])
        Regs.push_back(CheckedReg);
    }

    // ... a safe address can be materialized, or
    if (auto NewAddrReg = BC.MIB->getMaterializedAddressRegForPtrAuth(Point))
      Regs.push_back(*NewAddrReg);

    // ... an address can be updated in a safe manner, producing the result
    // which is as trusted as the input address.
    if (auto DstAndSrc = BC.MIB->analyzeAddressArithmeticsForPtrAuth(Point)) {
      if (Cur.TrustedRegs[DstAndSrc->second])
        Regs.push_back(DstAndSrc->first);
    }

    return Regs;
  }

  SrcState computeNext(const MCInst &Point, const SrcState &Cur) {
    if (BC.MIB->isCFI(Point))
      return Cur;

    SrcStatePrinter P(BC);
    LLVM_DEBUG({
      dbgs() << "  SrcSafetyAnalysis::ComputeNext(";
      BC.InstPrinter->printInst(&Point, 0, "", *BC.STI, dbgs());
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
      return SrcState();
    }

    // First, compute various properties of the instruction, taking the state
    // before its execution into account, if necessary.

    BitVector Clobbered = getClobberedRegs(Point);
    SmallVector<MCPhysReg> NewSafeToDerefRegs =
        getRegsMadeSafeToDeref(Point, Cur);
    SmallVector<MCPhysReg> NewTrustedRegs = getRegsMadeTrusted(Point, Cur);

    // Ideally, being trusted is a strictly stronger property than being
    // safe-to-dereference. To simplify the computation of Next state, enforce
    // this for NewSafeToDerefRegs and NewTrustedRegs. Additionally, this
    // fixes the properly for "cumulative" register states in tricky cases
    // like the following:
    //
    //    ; LR is safe to dereference here
    //    mov   x16, x30  ; start of the sequence, LR is s-t-d right before
    //    xpaclri         ; clobbers LR, LR is not safe anymore
    //    cmp   x30, x16
    //    b.eq  1f        ; end of the sequence: LR is marked as trusted
    //    brk   0x1234
    //  1:
    //    ; at this point LR would be marked as trusted,
    //    ; but not safe-to-dereference
    //
    for (auto TrustedReg : NewTrustedRegs) {
      if (!is_contained(NewSafeToDerefRegs, TrustedReg))
        NewSafeToDerefRegs.push_back(TrustedReg);
    }

    // Then, compute the state after this instruction is executed.
    SrcState Next = Cur;

    Next.SafeToDerefRegs.reset(Clobbered);
    Next.TrustedRegs.reset(Clobbered);
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

    // Process new trusted registers.
    for (MCPhysReg TrustedReg : NewTrustedRegs)
      Next.TrustedRegs |= BC.MIB->getAliases(TrustedReg, /*OnlySmaller=*/true);

    LLVM_DEBUG({
      dbgs() << "    .. result: (";
      P.print(dbgs(), Next);
      dbgs() << ")\n";
    });

    return Next;
  }

public:
  std::vector<MCInstReference>
  getLastClobberingInsts(const MCInst &Inst, BinaryFunction &BF,
                         MCPhysReg ClobberedReg) const {
    const SrcState &S = getStateBefore(Inst);

    std::vector<MCInstReference> Result;
    for (const MCInst *Inst : lastWritingInsts(S, ClobberedReg)) {
      MCInstReference Ref = MCInstReference::get(Inst, BF);
      assert(!Ref.empty() && "Expected Inst to be found");
      Result.push_back(Ref);
    }
    return Result;
  }
};

class DataflowSrcSafetyAnalysis
    : public SrcSafetyAnalysis,
      public DataflowAnalysis<DataflowSrcSafetyAnalysis, SrcState,
                              /*Backward=*/false, SrcStatePrinter> {
  using DFParent = DataflowAnalysis<DataflowSrcSafetyAnalysis, SrcState, false,
                                    SrcStatePrinter>;
  friend DFParent;

  using SrcSafetyAnalysis::BC;
  using SrcSafetyAnalysis::computeNext;

  // Pessimistic initial state for basic blocks without any predecessors
  // (not needed for most functions, thus initialized lazily).
  SrcState PessimisticState;

public:
  DataflowSrcSafetyAnalysis(BinaryFunction &BF,
                            MCPlusBuilder::AllocatorIdTy AllocId,
                            ArrayRef<MCPhysReg> RegsToTrackInstsFor)
      : SrcSafetyAnalysis(BF, RegsToTrackInstsFor), DFParent(BF, AllocId) {}

  const SrcState &getStateBefore(const MCInst &Inst) const override {
    return DFParent::getStateBefore(Inst).get();
  }

  void run() override {
    for (BinaryBasicBlock &BB : Func) {
      if (auto CheckerInfo = BC.MIB->getAuthCheckedReg(BB)) {
        MCPhysReg CheckedReg = CheckerInfo->first;
        MCInst &FirstInst = *CheckerInfo->second;
        MCInst &LastInst = *BB.getLastNonPseudoInstr();
        LLVM_DEBUG({
          dbgs() << "Found pointer checking sequence in " << BB.getName()
                 << ":\n";
          traceReg(BC, "Checked register", CheckedReg);
          traceInst(BC, "First instruction", FirstInst);
          traceInst(BC, "Last instruction", LastInst);
        });
        (void)CheckedReg;
        (void)FirstInst;
        assert(llvm::any_of(BB, [&](MCInst &I) { return &I == &FirstInst; }) &&
               "Data-flow analysis expects the checker not to cross BBs");
        CheckerSequenceInfo[&LastInst] = *CheckerInfo;
      }
    }
    DFParent::run();
  }

protected:
  void preflight() {}

  SrcState getStartingStateAtBB(const BinaryBasicBlock &BB) {
    if (BB.isEntryPoint())
      return createEntryState();

    // If a basic block without any predecessors is found in an optimized code,
    // this likely means that some CFG edges were not detected. Pessimistically
    // assume any register that can ever be clobbered in this function to be
    // unsafe before this basic block.
    // Warn about this fact in FunctionAnalysis::findUnsafeUses(), as it likely
    // means imprecise CFG information.
    if (BB.pred_empty()) {
      if (PessimisticState.empty())
        PessimisticState = computePessimisticState(*BB.getParent());
      return PessimisticState;
    }

    return SrcState();
  }

  SrcState getStartingStateAtPoint(const MCInst &Point) { return SrcState(); }

  void doConfluence(SrcState &StateOut, const SrcState &StateIn) {
    SrcStatePrinter P(BC);
    LLVM_DEBUG({
      dbgs() << "  DataflowSrcSafetyAnalysis::Confluence(\n";
      dbgs() << "    State 1: ";
      P.print(dbgs(), StateOut);
      dbgs() << "\n";
      dbgs() << "    State 2: ";
      P.print(dbgs(), StateIn);
      dbgs() << ")\n";
    });

    StateOut.merge(StateIn);

    LLVM_DEBUG({
      dbgs() << "    merged state: ";
      P.print(dbgs(), StateOut);
      dbgs() << "\n";
    });
  }

  StringRef getAnnotationName() const { return "DataflowSrcSafetyAnalysis"; }
};

/// A helper base class for implementing a simplified counterpart of a dataflow
/// analysis for functions without CFG information.
template <typename StateTy> class CFGUnawareAnalysis {
  BinaryContext &BC;
  BinaryFunction &BF;
  MCPlusBuilder::AllocatorIdTy AllocId;
  unsigned StateAnnotationIndex;

  void cleanStateAnnotations() {
    for (auto &I : BF.instrs())
      BC.MIB->removeAnnotation(I.second, StateAnnotationIndex);
  }

protected:
  CFGUnawareAnalysis(BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocId,
                     StringRef AnnotationName)
      : BC(BF.getBinaryContext()), BF(BF), AllocId(AllocId) {
    StateAnnotationIndex = BC.MIB->getOrCreateAnnotationIndex(AnnotationName);
  }

  void setState(MCInst &Inst, const StateTy &S) {
    // Check if we need to remove an old annotation (this is the case if
    // this is the second, detailed run of the analysis).
    if (BC.MIB->hasAnnotation(Inst, StateAnnotationIndex))
      BC.MIB->removeAnnotation(Inst, StateAnnotationIndex);
    // Attach the state.
    BC.MIB->addAnnotation(Inst, StateAnnotationIndex, S, AllocId);
  }

  const StateTy &getState(const MCInst &Inst) const {
    return BC.MIB->getAnnotationAs<StateTy>(Inst, StateAnnotationIndex);
  }

  virtual ~CFGUnawareAnalysis() { cleanStateAnnotations(); }
};

// A simplified implementation of DataflowSrcSafetyAnalysis for functions
// lacking CFG information.
//
// Let assume the instructions can only be executed linearly unless there is
// a label to jump to - this should handle both directly jumping to a location
// encoded as an immediate operand of a branch instruction, as well as saving a
// branch destination somewhere and passing it to an indirect branch instruction
// later, provided no arithmetic is performed on the destination address:
//
//     ; good: the destination is directly encoded into the branch instruction
//     cbz x0, some_label
//
//     ; good: the branch destination is first stored and then used as-is
//     adr x1, some_label
//     br  x1
//
//     ; bad: some clever arithmetic is performed manually
//     adr x1, some_label
//     add x1, x1, #4
//     br  x1
//     ...
//   some_label:
//     ; pessimistically reset the state as we are unsure where we came from
//     ...
//     ret
//   JTI0:
//     .byte some_label - Ltmp0 ; computing offsets using labels may probably
//                                work too, provided enough information is
//                                retained by the assembler and linker
//
// Then, a function can be split into a number of disjoint contiguous sequences
// of instructions without labels in between. These sequences can be processed
// the same way basic blocks are processed by data-flow analysis, with the same
// pessimistic estimation of the initial state at the start of each sequence
// (except the first instruction of the function).
class CFGUnawareSrcSafetyAnalysis : public SrcSafetyAnalysis,
                                    public CFGUnawareAnalysis<SrcState> {
  using SrcSafetyAnalysis::BC;
  BinaryFunction &BF;

public:
  CFGUnawareSrcSafetyAnalysis(BinaryFunction &BF,
                              MCPlusBuilder::AllocatorIdTy AllocId,
                              ArrayRef<MCPhysReg> RegsToTrackInstsFor)
      : SrcSafetyAnalysis(BF, RegsToTrackInstsFor),
        CFGUnawareAnalysis(BF, AllocId, "CFGUnawareSrcSafetyAnalysis"), BF(BF) {
  }

  void run() override {
    const SrcState DefaultState = computePessimisticState(BF);
    SrcState S = createEntryState();
    for (auto &I : BF.instrs()) {
      MCInst &Inst = I.second;
      if (BC.MIB->isCFI(Inst))
        continue;

      // If there is a label before this instruction, it is possible that it
      // can be jumped-to, thus conservatively resetting S. As an exception,
      // let's ignore any labels at the beginning of the function, as at least
      // one label is expected there.
      if (BF.hasLabelAt(I.first) && &Inst != &BF.instrs().begin()->second) {
        LLVM_DEBUG({
          traceInst(BC, "Due to label, resetting the state before", Inst);
        });
        S = DefaultState;
      }

      // Attach the state *before* this instruction executes.
      setState(Inst, S);

      // Compute the state after this instruction executes.
      S = computeNext(Inst, S);
    }
  }

  const SrcState &getStateBefore(const MCInst &Inst) const override {
    return getState(Inst);
  }
};

std::shared_ptr<SrcSafetyAnalysis>
SrcSafetyAnalysis::create(BinaryFunction &BF,
                          MCPlusBuilder::AllocatorIdTy AllocId,
                          ArrayRef<MCPhysReg> RegsToTrackInstsFor) {
  if (BF.hasCFG())
    return std::make_shared<DataflowSrcSafetyAnalysis>(BF, AllocId,
                                                       RegsToTrackInstsFor);
  return std::make_shared<CFGUnawareSrcSafetyAnalysis>(BF, AllocId,
                                                       RegsToTrackInstsFor);
}

/// A state representing which registers are safe to be used as the destination
/// operand of an authentication instruction.
///
/// Similar to SrcState, it is the responsibility of the analysis to take
/// register aliasing into account.
///
/// Depending on the implementation (such as whether FEAT_FPAC is implemented
/// by an AArch64 CPU or not), it may be possible that an authentication
/// instruction returns an invalid pointer on failure instead of terminating
/// the program immediately (assuming the program will crash as soon as that
/// pointer is dereferenced). Since few bits are usually allocated for the PAC
/// field (such as less than 16 bits on a typical AArch64 system), an attacker
/// can try every possible signature and guess the correct one if there is a
/// gadget that tells whether the particular pointer has a correct signature
/// (a so called "authentication oracle"). For that reason, it should be
/// impossible for an attacker to test if a pointer is correctly signed -
/// either the program should be terminated on authentication failure or
/// the result of authentication should not be accessible to an attacker.
///
/// Considering the instructions in forward order as they are executed, a
/// restricted set of operations can be allowed on any register containing a
/// value derived from the result of an authentication instruction until that
/// value is checked not to contain the result of a failed authentication.
/// In DstSafetyAnalysis, these rules are adapted, so that the safety property
/// for a register is computed by iterating the instructions in backward order.
/// Then the resulting properties are used at authentication instruction sites
/// to check output registers and report the particular instruction if it writes
/// to an unsafe register.
///
/// Another approach would be to simulate the above rules as-is, iterating over
/// the instructions in forward direction. To make it possible to report the
/// particular instructions as oracles, this would probably require tracking
/// references to these instructions for each register currently containing
/// sensitive data.
///
/// In DstSafetyAnalysis, the source register Xn of an instruction Inst is safe
/// if at least one of the following is true:
/// * Inst checks if Xn contains the result of a successful authentication and
///   terminates the program on failure. Note that Inst can either naturally
///   dereference Xn (load, branch, return, etc. instructions) or be the first
///   instruction of an explicit checking sequence.
/// * Inst performs safe address arithmetic AND both source and result
///   registers, as well as any temporary registers, must be safe after
///   execution of Inst (temporaries are not used on AArch64 and thus not
///   currently supported/allowed).
///   See MCPlusBuilder::analyzeAddressArithmeticsForPtrAuth for the details.
/// * Inst fully overwrites Xn with a constant.
struct DstState {
  /// The set of registers whose values cannot be inspected by an attacker in
  /// a way usable as an authentication oracle. The results of authentication
  /// instructions should only be written to such registers.
  BitVector CannotEscapeUnchecked;

  /// A vector of sets, only used on the second analysis run.
  /// Each element in this vector represents one of the tracked registers.
  /// For each such register we track the set of first instructions that leak
  /// the authenticated pointer before it was checked. This is intended to
  /// provide clues on which instruction made the particular register unsafe.
  ///
  /// Please note that the mapping from MCPhysReg values to indexes in this
  /// vector is provided by RegsToTrackInstsFor field of DstSafetyAnalysis.
  std::vector<SetOfRelatedInsts> FirstInstLeakingReg;

  /// Constructs an empty state.
  DstState() {}

  DstState(unsigned NumRegs, unsigned NumRegsToTrack)
      : CannotEscapeUnchecked(NumRegs), FirstInstLeakingReg(NumRegsToTrack) {}

  DstState &merge(const DstState &StateIn) {
    if (StateIn.empty())
      return *this;
    if (empty())
      return (*this = StateIn);

    CannotEscapeUnchecked &= StateIn.CannotEscapeUnchecked;
    for (unsigned I = 0; I < FirstInstLeakingReg.size(); ++I)
      for (const MCInst *J : StateIn.FirstInstLeakingReg[I])
        FirstInstLeakingReg[I].insert(J);
    return *this;
  }

  /// Returns true if this object does not store state of any registers -
  /// neither safe, nor unsafe ones.
  bool empty() const { return CannotEscapeUnchecked.empty(); }

  bool operator==(const DstState &RHS) const {
    return CannotEscapeUnchecked == RHS.CannotEscapeUnchecked &&
           FirstInstLeakingReg == RHS.FirstInstLeakingReg;
  }
  bool operator!=(const DstState &RHS) const { return !((*this) == RHS); }
};

static raw_ostream &operator<<(raw_ostream &OS, const DstState &S) {
  OS << "dst-state<";
  if (S.empty()) {
    OS << "empty";
  } else {
    OS << "CannotEscapeUnchecked: " << S.CannotEscapeUnchecked << ", ";
    printInstsShort(OS, S.FirstInstLeakingReg);
  }
  OS << ">";
  return OS;
}

class DstStatePrinter {
public:
  void print(raw_ostream &OS, const DstState &S) const;
  explicit DstStatePrinter(const BinaryContext &BC) : BC(BC) {}

private:
  const BinaryContext &BC;
};

void DstStatePrinter::print(raw_ostream &OS, const DstState &S) const {
  RegStatePrinter RegStatePrinter(BC);
  OS << "dst-state<";
  if (S.empty()) {
    assert(S.CannotEscapeUnchecked.empty());
    assert(S.FirstInstLeakingReg.empty());
    OS << "empty";
  } else {
    OS << "CannotEscapeUnchecked: ";
    RegStatePrinter.print(OS, S.CannotEscapeUnchecked);
    OS << ", ";
    printInstsShort(OS, S.FirstInstLeakingReg);
  }
  OS << ">";
}

/// Computes which registers are safe to be written to by auth instructions.
///
/// This is the base class for two implementations: a dataflow-based analysis
/// which is intended to be used for most functions and a simplified CFG-unaware
/// version for functions without reconstructed CFG.
class DstSafetyAnalysis {
public:
  DstSafetyAnalysis(BinaryFunction &BF, ArrayRef<MCPhysReg> RegsToTrackInstsFor)
      : BC(BF.getBinaryContext()), NumRegs(BC.MRI->getNumRegs()),
        RegsToTrackInstsFor(RegsToTrackInstsFor) {}

  virtual ~DstSafetyAnalysis() {}

  static std::shared_ptr<DstSafetyAnalysis>
  create(BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocId,
         ArrayRef<MCPhysReg> RegsToTrackInstsFor);

  virtual void run() = 0;
  virtual const DstState &getStateAfter(const MCInst &Inst) const = 0;

protected:
  BinaryContext &BC;
  const unsigned NumRegs;

  const TrackedRegisters RegsToTrackInstsFor;

  /// Stores information about the detected instruction sequences emitted to
  /// check an authenticated pointer. Specifically, if such sequence is detected
  /// in a basic block, it maps the first instruction of that sequence to the
  /// register being checked.
  ///
  /// As the detection of such sequences requires iterating over the adjacent
  /// instructions, it should be done before calling computeNext(), which
  /// operates on separate instructions.
  DenseMap<const MCInst *, MCPhysReg> RegCheckedAt;

  SetOfRelatedInsts &firstLeakingInsts(DstState &S, MCPhysReg Reg) const {
    unsigned Index = RegsToTrackInstsFor.getIndex(Reg);
    return S.FirstInstLeakingReg[Index];
  }
  const SetOfRelatedInsts &firstLeakingInsts(const DstState &S,
                                             MCPhysReg Reg) const {
    unsigned Index = RegsToTrackInstsFor.getIndex(Reg);
    return S.FirstInstLeakingReg[Index];
  }

  /// Creates a state with all registers marked unsafe (not to be confused
  /// with empty state).
  DstState createUnsafeState() {
    return DstState(NumRegs, RegsToTrackInstsFor.getNumTrackedRegisters());
  }

  /// Returns the set of registers that can be leaked by this instruction.
  /// A register is considered leaked if it has any intersection with any
  /// register read by Inst. This is similar to how the set of clobbered
  /// registers is computed, but taking input operands instead of outputs.
  BitVector getLeakedRegs(const MCInst &Inst) const {
    BitVector Leaked(NumRegs);

    // Assume a call can read all registers.
    if (BC.MIB->isCall(Inst)) {
      Leaked.set();
      return Leaked;
    }

    // Compute the set of registers overlapping with any register used by
    // this instruction.

    const MCInstrDesc &Desc = BC.MII->get(Inst.getOpcode());

    for (MCPhysReg Reg : Desc.implicit_uses())
      Leaked |= BC.MIB->getAliases(Reg, /*OnlySmaller=*/false);

    for (const MCOperand &Op : BC.MIB->useOperands(Inst)) {
      if (Op.isReg())
        Leaked |= BC.MIB->getAliases(Op.getReg(), /*OnlySmaller=*/false);
    }

    return Leaked;
  }

  SmallVector<MCPhysReg> getRegsMadeProtected(const MCInst &Inst,
                                              const BitVector &LeakedRegs,
                                              const DstState &Cur) const {
    SmallVector<MCPhysReg> Regs;

    // A pointer can be checked, or
    if (auto CheckedReg =
            BC.MIB->getAuthCheckedReg(Inst, /*MayOverwrite=*/true))
      Regs.push_back(*CheckedReg);
    if (RegCheckedAt.contains(&Inst))
      Regs.push_back(RegCheckedAt.at(&Inst));

    // ... it can be used as a branch target, or
    if (BC.MIB->isIndirectBranch(Inst) || BC.MIB->isIndirectCall(Inst)) {
      bool IsAuthenticated;
      MCPhysReg BranchDestReg =
          BC.MIB->getRegUsedAsIndirectBranchDest(Inst, IsAuthenticated);
      assert(BranchDestReg != BC.MIB->getNoRegister());
      if (!IsAuthenticated)
        Regs.push_back(BranchDestReg);
    }

    // ... it can be used as a return target, or
    if (BC.MIB->isReturn(Inst)) {
      bool IsAuthenticated = false;
      std::optional<MCPhysReg> RetReg =
          BC.MIB->getRegUsedAsRetDest(Inst, IsAuthenticated);
      if (RetReg && !IsAuthenticated)
        Regs.push_back(*RetReg);
    }

    // ... an address can be updated in a safe manner, or
    if (auto DstAndSrc = BC.MIB->analyzeAddressArithmeticsForPtrAuth(Inst)) {
      MCPhysReg DstReg, SrcReg;
      std::tie(DstReg, SrcReg) = *DstAndSrc;
      // Note that *all* registers containing the derived values must be safe,
      // both source and destination ones. No temporaries are supported at now.
      if (Cur.CannotEscapeUnchecked[SrcReg] &&
          Cur.CannotEscapeUnchecked[DstReg])
        Regs.push_back(SrcReg);
    }

    // ... the register can be overwritten in whole with a constant: for that
    // purpose, look for the instructions with no register inputs (neither
    // explicit nor implicit ones) and no side effects (to rule out reading
    // not modelled locations).
    const MCInstrDesc &Desc = BC.MII->get(Inst.getOpcode());
    bool HasExplicitSrcRegs = llvm::any_of(BC.MIB->useOperands(Inst),
                                           [](auto Op) { return Op.isReg(); });
    if (!Desc.hasUnmodeledSideEffects() && !HasExplicitSrcRegs &&
        Desc.implicit_uses().empty()) {
      for (const MCOperand &Def : BC.MIB->defOperands(Inst))
        Regs.push_back(Def.getReg());
    }

    return Regs;
  }

  DstState computeNext(const MCInst &Point, const DstState &Cur) {
    if (BC.MIB->isCFI(Point))
      return Cur;

    DstStatePrinter P(BC);
    LLVM_DEBUG({
      dbgs() << "  DstSafetyAnalysis::ComputeNext(";
      BC.InstPrinter->printInst(&Point, 0, "", *BC.STI, dbgs());
      dbgs() << ", ";
      P.print(dbgs(), Cur);
      dbgs() << ")\n";
    });

    // If this instruction terminates the program immediately, no
    // authentication oracles are possible past this point.
    if (BC.MIB->isTrap(Point)) {
      LLVM_DEBUG({ traceInst(BC, "Trap instruction found", Point); });
      DstState Next(NumRegs, RegsToTrackInstsFor.getNumTrackedRegisters());
      Next.CannotEscapeUnchecked.set();
      return Next;
    }

    // If this instruction is reachable by the analysis, a non-empty state will
    // be propagated to it sooner or later. Until then, skip computeNext().
    if (Cur.empty()) {
      LLVM_DEBUG(
          { dbgs() << "Skipping computeNext(Point, Cur) as Cur is empty.\n"; });
      return DstState();
    }

    // First, compute various properties of the instruction, taking the state
    // after its execution into account, if necessary.

    BitVector LeakedRegs = getLeakedRegs(Point);
    SmallVector<MCPhysReg> NewProtectedRegs =
        getRegsMadeProtected(Point, LeakedRegs, Cur);

    // Then, compute the state before this instruction is executed.
    DstState Next = Cur;

    Next.CannotEscapeUnchecked.reset(LeakedRegs);
    for (MCPhysReg Reg : RegsToTrackInstsFor.getRegisters()) {
      if (LeakedRegs[Reg])
        firstLeakingInsts(Next, Reg) = {&Point};
    }

    BitVector NewProtectedSubregs(NumRegs);
    for (MCPhysReg Reg : NewProtectedRegs)
      NewProtectedSubregs |= BC.MIB->getAliases(Reg, /*OnlySmaller=*/true);
    Next.CannotEscapeUnchecked |= NewProtectedSubregs;
    for (MCPhysReg Reg : RegsToTrackInstsFor.getRegisters()) {
      if (NewProtectedSubregs[Reg])
        firstLeakingInsts(Next, Reg).clear();
    }

    LLVM_DEBUG({
      dbgs() << "    .. result: (";
      P.print(dbgs(), Next);
      dbgs() << ")\n";
    });

    return Next;
  }

public:
  std::vector<MCInstReference> getLeakingInsts(const MCInst &Inst,
                                               BinaryFunction &BF,
                                               MCPhysReg LeakedReg) const {
    const DstState &S = getStateAfter(Inst);

    std::vector<MCInstReference> Result;
    for (const MCInst *Inst : firstLeakingInsts(S, LeakedReg)) {
      MCInstReference Ref = MCInstReference::get(Inst, BF);
      assert(!Ref.empty() && "Expected Inst to be found");
      Result.push_back(Ref);
    }
    return Result;
  }
};

class DataflowDstSafetyAnalysis
    : public DstSafetyAnalysis,
      public DataflowAnalysis<DataflowDstSafetyAnalysis, DstState,
                              /*Backward=*/true, DstStatePrinter> {
  using DFParent = DataflowAnalysis<DataflowDstSafetyAnalysis, DstState, true,
                                    DstStatePrinter>;
  friend DFParent;

  using DstSafetyAnalysis::BC;
  using DstSafetyAnalysis::computeNext;

public:
  DataflowDstSafetyAnalysis(BinaryFunction &BF,
                            MCPlusBuilder::AllocatorIdTy AllocId,
                            ArrayRef<MCPhysReg> RegsToTrackInstsFor)
      : DstSafetyAnalysis(BF, RegsToTrackInstsFor), DFParent(BF, AllocId) {}

  const DstState &getStateAfter(const MCInst &Inst) const override {
    // The dataflow analysis base class iterates backwards over the
    // instructions, thus "after" vs. "before" difference.
    return DFParent::getStateBefore(Inst).get();
  }

  void run() override {
    for (BinaryBasicBlock &BB : Func) {
      if (auto CheckerInfo = BC.MIB->getAuthCheckedReg(BB)) {
        LLVM_DEBUG({
          dbgs() << "Found pointer checking sequence in " << BB.getName()
                 << ":\n";
          traceReg(BC, "Checked register", CheckerInfo->first);
          traceInst(BC, "First instruction", *CheckerInfo->second);
        });
        RegCheckedAt[CheckerInfo->second] = CheckerInfo->first;
      }
    }
    DFParent::run();
  }

protected:
  void preflight() {}

  DstState getStartingStateAtBB(const BinaryBasicBlock &BB) {
    // In general, the initial state should be empty, not everything-is-unsafe,
    // to give a chance for some meaningful state to be propagated to BB from
    // an indirectly reachable "exit basic block" ending with a return or tail
    // call instruction.
    //
    // A basic block without any successors, on the other hand, can be
    // pessimistically initialized to everything-is-unsafe: this will naturally
    // handle return, trap and tail call instructions. At the same time, it is
    // harmless for internal indirect branch instructions, like computed gotos.
    if (BB.succ_empty())
      return createUnsafeState();

    return DstState();
  }

  DstState getStartingStateAtPoint(const MCInst &Point) { return DstState(); }

  void doConfluence(DstState &StateOut, const DstState &StateIn) {
    DstStatePrinter P(BC);
    LLVM_DEBUG({
      dbgs() << "  DataflowDstSafetyAnalysis::Confluence(\n";
      dbgs() << "    State 1: ";
      P.print(dbgs(), StateOut);
      dbgs() << "\n";
      dbgs() << "    State 2: ";
      P.print(dbgs(), StateIn);
      dbgs() << ")\n";
    });

    StateOut.merge(StateIn);

    LLVM_DEBUG({
      dbgs() << "    merged state: ";
      P.print(dbgs(), StateOut);
      dbgs() << "\n";
    });
  }

  StringRef getAnnotationName() const { return "DataflowDstSafetyAnalysis"; }
};

class CFGUnawareDstSafetyAnalysis : public DstSafetyAnalysis,
                                    public CFGUnawareAnalysis<DstState> {
  using DstSafetyAnalysis::BC;
  BinaryFunction &BF;

public:
  CFGUnawareDstSafetyAnalysis(BinaryFunction &BF,
                              MCPlusBuilder::AllocatorIdTy AllocId,
                              ArrayRef<MCPhysReg> RegsToTrackInstsFor)
      : DstSafetyAnalysis(BF, RegsToTrackInstsFor),
        CFGUnawareAnalysis(BF, AllocId, "CFGUnawareDstSafetyAnalysis"), BF(BF) {
  }

  void run() override {
    DstState S = createUnsafeState();
    for (auto &I : llvm::reverse(BF.instrs())) {
      MCInst &Inst = I.second;
      if (BC.MIB->isCFI(Inst))
        continue;

      // If Inst can change the control flow, we cannot be sure that the next
      // instruction (to be executed in analyzed program) is the one processed
      // on the previous iteration, thus pessimistically reset S before
      // starting to analyze Inst.
      if (BC.MIB->isCall(Inst) || BC.MIB->isBranch(Inst) ||
          BC.MIB->isReturn(Inst)) {
        LLVM_DEBUG({ traceInst(BC, "Control flow instruction", Inst); });
        S = createUnsafeState();
      }

      // Attach the state *after* this instruction executes.
      setState(Inst, S);

      // Compute the next state.
      S = computeNext(Inst, S);
    }
  }

  const DstState &getStateAfter(const MCInst &Inst) const override {
    return getState(Inst);
  }
};

std::shared_ptr<DstSafetyAnalysis>
DstSafetyAnalysis::create(BinaryFunction &BF,
                          MCPlusBuilder::AllocatorIdTy AllocId,
                          ArrayRef<MCPhysReg> RegsToTrackInstsFor) {
  if (BF.hasCFG())
    return std::make_shared<DataflowDstSafetyAnalysis>(BF, AllocId,
                                                       RegsToTrackInstsFor);
  return std::make_shared<CFGUnawareDstSafetyAnalysis>(BF, AllocId,
                                                       RegsToTrackInstsFor);
}

// This function could return PartialReport<T>, but currently T is always
// MCPhysReg, even though it is an implementation detail.
static PartialReport<MCPhysReg> make_generic_report(MCInstReference Location,
                                                    StringRef Text) {
  auto Report = std::make_shared<GenericDiagnostic>(Location, Text);
  return PartialReport<MCPhysReg>(Report, std::nullopt);
}

template <typename T>
static PartialReport<T> make_gadget_report(const GadgetKind &Kind,
                                           MCInstReference Location,
                                           T RequestedDetails) {
  auto Report = std::make_shared<GadgetDiagnostic>(Kind, Location);
  return PartialReport<T>(Report, RequestedDetails);
}

static std::optional<PartialReport<MCPhysReg>>
shouldReportReturnGadget(const BinaryContext &BC, const MCInstReference &Inst,
                         const SrcState &S) {
  static const GadgetKind RetKind("non-protected ret found");
  if (!BC.MIB->isReturn(Inst))
    return std::nullopt;

  bool IsAuthenticated = false;
  std::optional<MCPhysReg> RetReg =
      BC.MIB->getRegUsedAsRetDest(Inst, IsAuthenticated);
  if (!RetReg) {
    return make_generic_report(
        Inst, "Warning: pac-ret analysis could not analyze this return "
              "instruction");
  }
  if (IsAuthenticated)
    return std::nullopt;

  LLVM_DEBUG({
    traceInst(BC, "Found RET inst", Inst);
    traceReg(BC, "RetReg", *RetReg);
    traceRegMask(BC, "SafeToDerefRegs", S.SafeToDerefRegs);
  });

  if (S.SafeToDerefRegs[*RetReg])
    return std::nullopt;

  return make_gadget_report(RetKind, Inst, *RetReg);
}

/// While BOLT already marks some of the branch instructions as tail calls,
/// this function tries to detect less obvious cases, assuming false positives
/// are acceptable as long as there are not too many of them.
///
/// It is possible that not all the instructions classified as tail calls by
/// this function are safe to be considered as such for the purpose of code
/// transformations performed by BOLT. The intention of this function is to
/// spot some of actually missed tail calls (and likely a number of unrelated
/// indirect branch instructions) as long as this doesn't increase the amount
/// of false positive reports unacceptably.
static bool shouldAnalyzeTailCallInst(const BinaryContext &BC,
                                      const BinaryFunction &BF,
                                      const MCInstReference &Inst) {
  // Some BC.MIB->isXYZ(Inst) methods simply delegate to MCInstrDesc::isXYZ()
  // (such as isBranch at the time of writing this comment), some don't (such
  // as isCall). For that reason, call MCInstrDesc's methods explicitly when
  // it is important.
  const MCInstrDesc &Desc = BC.MII->get(Inst.getMCInst().getOpcode());
  // Tail call should be a branch (but not necessarily an indirect one).
  if (!Desc.isBranch())
    return false;

  // Always analyze the branches already marked as tail calls by BOLT.
  if (BC.MIB->isTailCall(Inst))
    return true;

  // Try to also check the branches marked as "UNKNOWN CONTROL FLOW" - the
  // below is a simplified condition from BinaryContext::printInstruction.
  bool IsUnknownControlFlow =
      BC.MIB->isIndirectBranch(Inst) && !BC.MIB->getJumpTable(Inst);

  if (BF.hasCFG() && IsUnknownControlFlow)
    return true;

  return false;
}

static std::optional<PartialReport<MCPhysReg>>
shouldReportUnsafeTailCall(const BinaryContext &BC, const BinaryFunction &BF,
                           const MCInstReference &Inst, const SrcState &S) {
  static const GadgetKind UntrustedLRKind(
      "untrusted link register found before tail call");

  if (!shouldAnalyzeTailCallInst(BC, BF, Inst))
    return std::nullopt;

  // Not only the set of registers returned by getTrustedLiveInRegs() can be
  // seen as a reasonable target-independent _approximation_ of "the LR", these
  // are *exactly* those registers used by SrcSafetyAnalysis to initialize the
  // set of trusted registers on function entry.
  // Thus, this function basically checks that the precondition expected to be
  // imposed by a function call instruction (which is hardcoded into the target-
  // specific getTrustedLiveInRegs() function) is also respected on tail calls.
  SmallVector<MCPhysReg> RegsToCheck = BC.MIB->getTrustedLiveInRegs();
  LLVM_DEBUG({
    traceInst(BC, "Found tail call inst", Inst);
    traceRegMask(BC, "Trusted regs", S.TrustedRegs);
  });

  // In musl on AArch64, the _start function sets LR to zero and calls the next
  // stage initialization function at the end, something along these lines:
  //
  //   _start:
  //     mov     x30, #0
  //     ; ... other initialization ...
  //     b       _start_c ; performs "exit" system call at some point
  //
  // As this would produce a false positive for every executable linked with
  // such libc, ignore tail calls performed by ELF entry function.
  if (BC.StartFunctionAddress &&
      *BC.StartFunctionAddress == Inst.getFunction()->getAddress()) {
    LLVM_DEBUG({ dbgs() << "  Skipping tail call in ELF entry function.\n"; });
    return std::nullopt;
  }

  // Returns at most one report per instruction - this is probably OK...
  for (auto Reg : RegsToCheck)
    if (!S.TrustedRegs[Reg])
      return make_gadget_report(UntrustedLRKind, Inst, Reg);

  return std::nullopt;
}

static std::optional<PartialReport<MCPhysReg>>
shouldReportCallGadget(const BinaryContext &BC, const MCInstReference &Inst,
                       const SrcState &S) {
  static const GadgetKind CallKind("non-protected call found");
  if (!BC.MIB->isIndirectCall(Inst) && !BC.MIB->isIndirectBranch(Inst))
    return std::nullopt;

  bool IsAuthenticated = false;
  MCPhysReg DestReg =
      BC.MIB->getRegUsedAsIndirectBranchDest(Inst, IsAuthenticated);
  if (IsAuthenticated)
    return std::nullopt;

  assert(DestReg != BC.MIB->getNoRegister() && "Valid register expected");
  LLVM_DEBUG({
    traceInst(BC, "Found call inst", Inst);
    traceReg(BC, "Call destination reg", DestReg);
    traceRegMask(BC, "SafeToDerefRegs", S.SafeToDerefRegs);
  });
  if (S.SafeToDerefRegs[DestReg])
    return std::nullopt;

  return make_gadget_report(CallKind, Inst, DestReg);
}

static std::optional<PartialReport<MCPhysReg>>
shouldReportSigningOracle(const BinaryContext &BC, const MCInstReference &Inst,
                          const SrcState &S) {
  static const GadgetKind SigningOracleKind("signing oracle found");

  std::optional<MCPhysReg> SignedReg = BC.MIB->getSignedReg(Inst);
  if (!SignedReg)
    return std::nullopt;

  LLVM_DEBUG({
    traceInst(BC, "Found sign inst", Inst);
    traceReg(BC, "Signed reg", *SignedReg);
    traceRegMask(BC, "TrustedRegs", S.TrustedRegs);
  });
  if (S.TrustedRegs[*SignedReg])
    return std::nullopt;

  return make_gadget_report(SigningOracleKind, Inst, *SignedReg);
}

static std::optional<PartialReport<MCPhysReg>>
shouldReportAuthOracle(const BinaryContext &BC, const MCInstReference &Inst,
                       const DstState &S) {
  static const GadgetKind AuthOracleKind("authentication oracle found");

  bool IsChecked = false;
  std::optional<MCPhysReg> AuthReg =
      BC.MIB->getWrittenAuthenticatedReg(Inst, IsChecked);
  if (!AuthReg || IsChecked)
    return std::nullopt;

  LLVM_DEBUG({
    traceInst(BC, "Found auth inst", Inst);
    traceReg(BC, "Authenticated reg", *AuthReg);
  });

  if (S.empty()) {
    LLVM_DEBUG({ dbgs() << "    DstState is empty!\n"; });
    return make_generic_report(
        Inst, "Warning: no state computed for an authentication instruction "
              "(possibly unreachable)");
  }

  LLVM_DEBUG(
      { traceRegMask(BC, "safe output registers", S.CannotEscapeUnchecked); });
  if (S.CannotEscapeUnchecked[*AuthReg])
    return std::nullopt;

  return make_gadget_report(AuthOracleKind, Inst, *AuthReg);
}

static SmallVector<MCPhysReg>
collectRegsToTrack(ArrayRef<PartialReport<MCPhysReg>> Reports) {
  SmallSet<MCPhysReg, 4> RegsToTrack;
  for (auto Report : Reports)
    if (Report.RequestedDetails)
      RegsToTrack.insert(*Report.RequestedDetails);

  return SmallVector<MCPhysReg>(RegsToTrack.begin(), RegsToTrack.end());
}

void FunctionAnalysisContext::findUnsafeUses(
    SmallVector<PartialReport<MCPhysReg>> &Reports) {
  auto Analysis = SrcSafetyAnalysis::create(BF, AllocatorId, {});
  LLVM_DEBUG({ dbgs() << "Running src register safety analysis...\n"; });
  Analysis->run();
  LLVM_DEBUG({
    dbgs() << "After src register safety analysis:\n";
    BF.dump();
  });

  bool UnreachableBBReported = false;
  if (BF.hasCFG()) {
    // Warn on basic blocks being unreachable according to BOLT (at most once
    // per BinaryFunction), as this likely means the CFG reconstructed by BOLT
    // is imprecise. A basic block can be
    // * reachable from an entry basic block - a hopefully correct non-empty
    //   state is propagated to that basic block sooner or later. All basic
    //   blocks are expected to belong to this category under normal conditions.
    // * reachable from a "directly unreachable" BB (a basic block that has no
    //   direct predecessors and this is not because it is an entry BB) - *some*
    //   non-empty state is propagated to this basic block sooner or later, as
    //   the initial state of directly unreachable basic blocks is
    //   pessimistically initialized to "all registers are unsafe"
    //   - a warning can be printed for the "directly unreachable" basic block
    // * neither reachable from an entry nor from a "directly unreachable" BB
    //   (such as if this BB is in an isolated loop of basic blocks) - the final
    //   state is computed to be empty for this basic block
    //   - a warning can be printed for this basic block
    for (BinaryBasicBlock &BB : BF) {
      MCInst *FirstInst = BB.getFirstNonPseudoInstr();
      // Skip empty basic block early for simplicity.
      if (!FirstInst)
        continue;

      bool IsDirectlyUnreachable = BB.pred_empty() && !BB.isEntryPoint();
      bool HasNoStateComputed = Analysis->getStateBefore(*FirstInst).empty();
      if (!IsDirectlyUnreachable && !HasNoStateComputed)
        continue;

      // Arbitrarily attach the report to the first instruction of BB.
      // This is printed as "[message] in function [name], basic block ...,
      // at address ..." when the issue is reported to the user.
      Reports.push_back(make_generic_report(
          MCInstReference(&BB, FirstInst),
          "Warning: possibly imprecise CFG, the analysis quality may be "
          "degraded in this function. According to BOLT, unreachable code is "
          "found" /* in function [name]... */));
      UnreachableBBReported = true;
      break; // One warning per function.
    }
  }
  // FIXME: Warn the user about imprecise analysis when the function has no CFG
  //        information at all.

  iterateOverInstrs(BF, [&](MCInstReference Inst) {
    if (BC.MIB->isCFI(Inst))
      return;

    const SrcState &S = Analysis->getStateBefore(Inst);
    if (S.empty()) {
      LLVM_DEBUG(
          { traceInst(BC, "Instruction has no state, skipping", Inst); });
      assert(UnreachableBBReported && "Should be reported at least once");
      (void)UnreachableBBReported;
      return;
    }

    if (auto Report = shouldReportReturnGadget(BC, Inst, S))
      Reports.push_back(*Report);

    if (PacRetGadgetsOnly)
      return;

    if (auto Report = shouldReportUnsafeTailCall(BC, BF, Inst, S))
      Reports.push_back(*Report);

    if (auto Report = shouldReportCallGadget(BC, Inst, S))
      Reports.push_back(*Report);
    if (auto Report = shouldReportSigningOracle(BC, Inst, S))
      Reports.push_back(*Report);
  });
}

void FunctionAnalysisContext::augmentUnsafeUseReports(
    ArrayRef<PartialReport<MCPhysReg>> Reports) {
  SmallVector<MCPhysReg> RegsToTrack = collectRegsToTrack(Reports);
  // Re-compute the analysis with register tracking.
  auto Analysis = SrcSafetyAnalysis::create(BF, AllocatorId, RegsToTrack);
  LLVM_DEBUG(
      { dbgs() << "\nRunning detailed src register safety analysis...\n"; });
  Analysis->run();
  LLVM_DEBUG({
    dbgs() << "After detailed src register safety analysis:\n";
    BF.dump();
  });

  // Augment gadget reports.
  for (auto &Report : Reports) {
    MCInstReference Location = Report.Issue->Location;
    LLVM_DEBUG({ traceInst(BC, "Attaching clobbering info to", Location); });
    assert(Report.RequestedDetails &&
           "Should be removed by handleSimpleReports");
    auto DetailedInfo =
        std::make_shared<ClobberingInfo>(Analysis->getLastClobberingInsts(
            Location, BF, *Report.RequestedDetails));
    Result.Diagnostics.emplace_back(Report.Issue, DetailedInfo);
  }
}

void FunctionAnalysisContext::findUnsafeDefs(
    SmallVector<PartialReport<MCPhysReg>> &Reports) {
  if (PacRetGadgetsOnly)
    return;

  auto Analysis = DstSafetyAnalysis::create(BF, AllocatorId, {});
  LLVM_DEBUG({ dbgs() << "Running dst register safety analysis...\n"; });
  Analysis->run();
  LLVM_DEBUG({
    dbgs() << "After dst register safety analysis:\n";
    BF.dump();
  });

  iterateOverInstrs(BF, [&](MCInstReference Inst) {
    if (BC.MIB->isCFI(Inst))
      return;

    const DstState &S = Analysis->getStateAfter(Inst);

    if (auto Report = shouldReportAuthOracle(BC, Inst, S))
      Reports.push_back(*Report);
  });
}

void FunctionAnalysisContext::augmentUnsafeDefReports(
    ArrayRef<PartialReport<MCPhysReg>> Reports) {
  SmallVector<MCPhysReg> RegsToTrack = collectRegsToTrack(Reports);
  // Re-compute the analysis with register tracking.
  auto Analysis = DstSafetyAnalysis::create(BF, AllocatorId, RegsToTrack);
  LLVM_DEBUG(
      { dbgs() << "\nRunning detailed dst register safety analysis...\n"; });
  Analysis->run();
  LLVM_DEBUG({
    dbgs() << "After detailed dst register safety analysis:\n";
    BF.dump();
  });

  // Augment gadget reports.
  for (auto &Report : Reports) {
    MCInstReference Location = Report.Issue->Location;
    LLVM_DEBUG({ traceInst(BC, "Attaching leakage info to", Location); });
    assert(Report.RequestedDetails &&
           "Should be removed by handleSimpleReports");
    auto DetailedInfo = std::make_shared<LeakageInfo>(
        Analysis->getLeakingInsts(Location, BF, *Report.RequestedDetails));
    Result.Diagnostics.emplace_back(Report.Issue, DetailedInfo);
  }
}

void FunctionAnalysisContext::handleSimpleReports(
    SmallVector<PartialReport<MCPhysReg>> &Reports) {
  // Before re-running the detailed analysis, process the reports which do not
  // need any additional details to be attached.
  for (auto &Report : Reports) {
    if (!Report.RequestedDetails)
      Result.Diagnostics.emplace_back(Report.Issue, nullptr);
  }
  llvm::erase_if(Reports, [](const auto &R) { return !R.RequestedDetails; });
}

void FunctionAnalysisContext::run() {
  LLVM_DEBUG({
    dbgs() << "Analyzing function " << BF.getPrintName()
           << ", AllocatorId = " << AllocatorId << "\n";
    BF.dump();
  });

  SmallVector<PartialReport<MCPhysReg>> UnsafeUses;
  findUnsafeUses(UnsafeUses);
  handleSimpleReports(UnsafeUses);
  if (!UnsafeUses.empty())
    augmentUnsafeUseReports(UnsafeUses);

  SmallVector<PartialReport<MCPhysReg>> UnsafeDefs;
  findUnsafeDefs(UnsafeDefs);
  handleSimpleReports(UnsafeDefs);
  if (!UnsafeDefs.empty())
    augmentUnsafeDefReports(UnsafeDefs);
}

void Analysis::runOnFunction(BinaryFunction &BF,
                             MCPlusBuilder::AllocatorIdTy AllocatorId) {
  FunctionAnalysisContext FA(BF, AllocatorId, PacRetGadgetsOnly);
  FA.run();

  const FunctionAnalysisResult &FAR = FA.getResult();
  if (FAR.Diagnostics.empty())
    return;

  // `runOnFunction` is typically getting called from multiple threads in
  // parallel. Therefore, use a lock to avoid data races when storing the
  // result of the analysis in the `AnalysisResults` map.
  {
    std::lock_guard<std::mutex> Lock(AnalysisResultsMutex);
    AnalysisResults[&BF] = FAR;
  }
}

// Compute the instruction address for printing (may be slow).
static uint64_t getAddress(const MCInstReference &Inst) {
  const BinaryFunction *BF = Inst.getFunction();

  if (Inst.hasCFG()) {
    const BinaryBasicBlock *BB = Inst.getBasicBlock();

    auto It = static_cast<BinaryBasicBlock::const_iterator>(&Inst.getMCInst());
    unsigned IndexInBB = std::distance(BB->begin(), It);

    // FIXME: this assumes all instructions are 4 bytes in size. This is true
    // for AArch64, but it might be good to extract this function so it can be
    // used elsewhere and for other targets too.
    return BF->getAddress() + BB->getOffset() + IndexInBB * 4;
  }

  for (auto I = BF->instrs().begin(), E = BF->instrs().end(); I != E; ++I) {
    if (&I->second == &Inst.getMCInst())
      return BF->getAddress() + I->first;
  }
  llvm_unreachable("Instruction not found in function");
}

static void printBB(const BinaryContext &BC, const BinaryBasicBlock *BB,
                    size_t StartIndex = 0, size_t EndIndex = -1) {
  if (EndIndex == (size_t)-1)
    EndIndex = BB->size() - 1;
  const BinaryFunction *BF = BB->getFunction();
  for (unsigned I = StartIndex; I <= EndIndex; ++I) {
    MCInstReference Inst(BB, I);
    if (BC.MIB->isCFI(Inst))
      continue;
    BC.printInstruction(outs(), Inst, getAddress(Inst), BF);
  }
}

static void reportFoundGadgetInSingleBBSingleRelatedInst(
    raw_ostream &OS, const BinaryContext &BC, const MCInstReference RelatedInst,
    const MCInstReference Location) {
  const BinaryBasicBlock *BB = Location.getBasicBlock();
  assert(RelatedInst.hasCFG());
  assert(Location.hasCFG());
  if (BB == RelatedInst.getBasicBlock()) {
    OS << "  This happens in the following basic block:\n";
    printBB(BC, BB);
  }
}

void Diagnostic::printBasicInfo(raw_ostream &OS, const BinaryContext &BC,
                                StringRef IssueKind) const {
  const BinaryBasicBlock *BB = Location.getBasicBlock();
  const BinaryFunction *BF = Location.getFunction();

  OS << "\nGS-PAUTH: " << IssueKind;
  OS << " in function " << BF->getPrintName();
  if (BB)
    OS << ", basic block " << BB->getName();
  OS << ", at address " << llvm::format("%x", getAddress(Location)) << "\n";
  OS << "  The instruction is ";
  BC.printInstruction(OS, Location, getAddress(Location), BF);
}

void GadgetDiagnostic::generateReport(raw_ostream &OS,
                                      const BinaryContext &BC) const {
  printBasicInfo(OS, BC, Kind.getDescription());
}

static void printRelatedInstrs(raw_ostream &OS, const MCInstReference Location,
                               ArrayRef<MCInstReference> RelatedInstrs) {
  const BinaryFunction &BF = *Location.getFunction();
  const BinaryContext &BC = BF.getBinaryContext();

  // Sort by address to ensure output is deterministic.
  SmallVector<MCInstReference> RI(RelatedInstrs);
  llvm::sort(RI, [](const MCInstReference &A, const MCInstReference &B) {
    return getAddress(A) < getAddress(B);
  });
  for (unsigned I = 0; I < RI.size(); ++I) {
    MCInstReference InstRef = RI[I];
    OS << "  " << (I + 1) << ". ";
    BC.printInstruction(OS, InstRef, getAddress(InstRef), &BF);
  };
  if (RelatedInstrs.size() == 1) {
    const MCInstReference RelatedInst = RelatedInstrs[0];
    // Printing the details for the MCInstReference::FunctionParent case
    // is not implemented not to overcomplicate the code, as most functions
    // are expected to have CFG information.
    if (RelatedInst.hasCFG())
      reportFoundGadgetInSingleBBSingleRelatedInst(OS, BC, RelatedInst,
                                                   Location);
  }
}

void ClobberingInfo::print(raw_ostream &OS,
                           const MCInstReference Location) const {
  OS << "  The " << ClobberingInstrs.size()
     << " instructions that write to the affected registers after any "
        "authentication are:\n";
  printRelatedInstrs(OS, Location, ClobberingInstrs);
}

void LeakageInfo::print(raw_ostream &OS, const MCInstReference Location) const {
  OS << "  The " << LeakingInstrs.size()
     << " instructions that leak the affected registers are:\n";
  printRelatedInstrs(OS, Location, LeakingInstrs);
}

void GenericDiagnostic::generateReport(raw_ostream &OS,
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

  for (BinaryFunction *BF : BC.getAllBinaryFunctions()) {
    if (!AnalysisResults.count(BF))
      continue;
    for (const FinalReport &R : AnalysisResults[BF].Diagnostics) {
      R.Issue->generateReport(outs(), BC);
      if (R.Details)
        R.Details->print(outs(), R.Issue->Location);
    }
  }
  return Error::success();
}

} // namespace PAuthGadgetScanner
} // namespace bolt
} // namespace llvm
