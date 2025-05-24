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
  std::vector<SmallPtrSet<const MCInst *, 4>> LastInstWritingReg;

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

static void
printLastInsts(raw_ostream &OS,
               ArrayRef<SmallPtrSet<const MCInst *, 4>> LastInstWritingReg) {
  OS << "Insts: ";
  for (unsigned I = 0; I < LastInstWritingReg.size(); ++I) {
    auto &Set = LastInstWritingReg[I];
    OS << "[" << I << "](";
    for (const MCInst *MCInstP : Set)
      OS << MCInstP << " ";
    OS << ")";
  }
}

raw_ostream &operator<<(raw_ostream &OS, const SrcState &S) {
  OS << "src-state<";
  if (S.empty()) {
    OS << "empty";
  } else {
    OS << "SafeToDerefRegs: " << S.SafeToDerefRegs << ", ";
    OS << "TrustedRegs: " << S.TrustedRegs << ", ";
    printLastInsts(OS, S.LastInstWritingReg);
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
    printLastInsts(OS, S.LastInstWritingReg);
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

  SmallPtrSet<const MCInst *, 4> &lastWritingInsts(SrcState &S,
                                                   MCPhysReg Reg) const {
    unsigned Index = RegsToTrackInstsFor.getIndex(Reg);
    return S.LastInstWritingReg[Index];
  }
  const SmallPtrSet<const MCInst *, 4> &lastWritingInsts(const SrcState &S,
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

  // Returns all registers made trusted by this instruction.
  SmallVector<MCPhysReg> getRegsMadeTrusted(const MCInst &Point,
                                            const SrcState &Cur) const {
    SmallVector<MCPhysReg> Regs;
    const MCPhysReg NoReg = BC.MIB->getNoRegister();

    // An authenticated pointer can be checked, or
    MCPhysReg CheckedReg =
        BC.MIB->getAuthCheckedReg(Point, /*MayOverwrite=*/false);
    if (CheckedReg != NoReg && Cur.SafeToDerefRegs[CheckedReg])
      Regs.push_back(CheckedReg);

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
    MCPhysReg NewAddrReg = BC.MIB->getMaterializedAddressRegForPtrAuth(Point);
    if (NewAddrReg != NoReg)
      Regs.push_back(NewAddrReg);

    // ... an address can be updated in a safe manner, producing the result
    // which is as trusted as the input address.
    if (auto DstAndSrc = BC.MIB->analyzeAddressArithmeticsForPtrAuth(Point)) {
      if (Cur.TrustedRegs[DstAndSrc->second])
        Regs.push_back(DstAndSrc->first);
    }

    return Regs;
  }

  SrcState computeNext(const MCInst &Point, const SrcState &Cur) {
    SrcStatePrinter P(BC);
    LLVM_DEBUG({
      dbgs() << "  SrcSafetyAnalysis::ComputeNext(";
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
      assert(Ref && "Expected Inst to be found");
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
// the same way basic blocks are processed by data-flow analysis, assuming
// pessimistically that all registers are unsafe at the start of each sequence.
class CFGUnawareSrcSafetyAnalysis : public SrcSafetyAnalysis {
  BinaryFunction &BF;
  MCPlusBuilder::AllocatorIdTy AllocId;
  unsigned StateAnnotationIndex;

  void cleanStateAnnotations() {
    for (auto &I : BF.instrs())
      BC.MIB->removeAnnotation(I.second, StateAnnotationIndex);
  }

  /// Creates a state with all registers marked unsafe (not to be confused
  /// with empty state).
  SrcState createUnsafeState() const {
    return SrcState(NumRegs, RegsToTrackInstsFor.getNumTrackedRegisters());
  }

public:
  CFGUnawareSrcSafetyAnalysis(BinaryFunction &BF,
                              MCPlusBuilder::AllocatorIdTy AllocId,
                              ArrayRef<MCPhysReg> RegsToTrackInstsFor)
      : SrcSafetyAnalysis(BF, RegsToTrackInstsFor), BF(BF), AllocId(AllocId) {
    StateAnnotationIndex =
        BC.MIB->getOrCreateAnnotationIndex("CFGUnawareSrcSafetyAnalysis");
  }

  void run() override {
    SrcState S = createEntryState();
    for (auto &I : BF.instrs()) {
      MCInst &Inst = I.second;

      // If there is a label before this instruction, it is possible that it
      // can be jumped-to, thus conservatively resetting S. As an exception,
      // let's ignore any labels at the beginning of the function, as at least
      // one label is expected there.
      if (BF.hasLabelAt(I.first) && &Inst != &BF.instrs().begin()->second) {
        LLVM_DEBUG({
          traceInst(BC, "Due to label, resetting the state before", Inst);
        });
        S = createUnsafeState();
      }

      // Check if we need to remove an old annotation (this is the case if
      // this is the second, detailed, run of the analysis).
      if (BC.MIB->hasAnnotation(Inst, StateAnnotationIndex))
        BC.MIB->removeAnnotation(Inst, StateAnnotationIndex);
      // Attach the state *before* this instruction executes.
      BC.MIB->addAnnotation(Inst, StateAnnotationIndex, S, AllocId);

      // Compute the state after this instruction executes.
      S = computeNext(Inst, S);
    }
  }

  const SrcState &getStateBefore(const MCInst &Inst) const override {
    return BC.MIB->getAnnotationAs<SrcState>(Inst, StateAnnotationIndex);
  }

  ~CFGUnawareSrcSafetyAnalysis() { cleanStateAnnotations(); }
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

  ErrorOr<MCPhysReg> MaybeRetReg = BC.MIB->getRegUsedAsRetDest(Inst);
  if (MaybeRetReg.getError()) {
    return make_generic_report(
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
    return std::nullopt;
  LLVM_DEBUG({ traceRegMask(BC, "SafeToDerefRegs", S.SafeToDerefRegs); });
  if (S.SafeToDerefRegs[RetReg])
    return std::nullopt;

  return make_gadget_report(RetKind, Inst, RetReg);
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

  assert(DestReg != BC.MIB->getNoRegister());
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

  MCPhysReg SignedReg = BC.MIB->getSignedReg(Inst);
  if (SignedReg == BC.MIB->getNoRegister())
    return std::nullopt;

  LLVM_DEBUG({
    traceInst(BC, "Found sign inst", Inst);
    traceReg(BC, "Signed reg", SignedReg);
    traceRegMask(BC, "TrustedRegs", S.TrustedRegs);
  });
  if (S.TrustedRegs[SignedReg])
    return std::nullopt;

  return make_gadget_report(SigningOracleKind, Inst, SignedReg);
}

template <typename T> static void iterateOverInstrs(BinaryFunction &BF, T Fn) {
  if (BF.hasCFG()) {
    for (BinaryBasicBlock &BB : BF)
      for (int64_t I = 0, E = BB.size(); I < E; ++I)
        Fn(MCInstInBBReference(&BB, I));
  } else {
    for (auto I : BF.instrs())
      Fn(MCInstInBFReference(&BF, I.first));
  }
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

  iterateOverInstrs(BF, [&](MCInstReference Inst) {
    const SrcState &S = Analysis->getStateBefore(Inst);

    // If non-empty state was never propagated from the entry basic block
    // to Inst, assume it to be unreachable and report a warning.
    if (S.empty()) {
      Reports.push_back(
          make_generic_report(Inst, "Warning: unreachable instruction found"));
      return;
    }

    if (auto Report = shouldReportReturnGadget(BC, Inst, S))
      Reports.push_back(*Report);

    if (PacRetGadgetsOnly)
      return;

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

static void reportFoundGadgetInSingleBBSingleRelatedInst(
    raw_ostream &OS, const BinaryContext &BC, const MCInstReference RelatedInst,
    const MCInstReference Location) {
  BinaryBasicBlock *BB = Location.getBasicBlock();
  assert(RelatedInst.ParentKind == MCInstReference::BasicBlockParent);
  assert(Location.ParentKind == MCInstReference::BasicBlockParent);
  MCInstInBBReference RelatedInstBB = RelatedInst.U.BBRef;
  if (BB == RelatedInstBB.BB) {
    OS << "  This happens in the following basic block:\n";
    printBB(BC, BB);
  }
}

void Diagnostic::printBasicInfo(raw_ostream &OS, const BinaryContext &BC,
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
    return A.getAddress() < B.getAddress();
  });
  for (unsigned I = 0; I < RI.size(); ++I) {
    MCInstReference InstRef = RI[I];
    OS << "  " << (I + 1) << ". ";
    BC.printInstruction(OS, InstRef, InstRef.getAddress(), &BF);
  };
  if (RelatedInstrs.size() == 1) {
    const MCInstReference RelatedInst = RelatedInstrs[0];
    // Printing the details for the MCInstReference::FunctionParent case
    // is not implemented not to overcomplicate the code, as most functions
    // are expected to have CFG information.
    if (RelatedInst.ParentKind == MCInstReference::BasicBlockParent)
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
