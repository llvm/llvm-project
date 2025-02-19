//===- bolt/Passes/NonPacProtectedRetAnalysis.cpp -------------------------===//
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

#include "bolt/Passes/NonPacProtectedRetAnalysis.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Passes/DataflowAnalysis.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Format.h"
#include <memory>

#define DEBUG_TYPE "bolt-nonpacprotectedret"

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

namespace NonPacProtectedRetAnalysis {

// The security property that is checked is:
// When a register is used as the address to jump to in a return instruction,
// that register must either:
// (a) never be changed within this function, i.e. have the same value as when
//     the function started, or
// (b) the last write to the register must be by an authentication instruction.

// This property is checked by using dataflow analysis to keep track of which
// registers have been written (def-ed), since last authenticated. Those are
// exactly the registers containing values that should not be trusted (as they
// could have changed since the last time they were authenticated). For pac-ret,
// any return instruction using such a register is a gadget to be reported. For
// PAuthABI, probably at least any indirect control flow using such a register
// should be reported.

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

struct State {
  /// A BitVector containing the registers that have been clobbered, and
  /// not authenticated.
  BitVector NonAutClobRegs;
  /// A vector of sets, only used in the second data flow run.
  /// Each element in the vector represents one of the registers for which we
  /// track the set of last instructions that wrote to this register. For
  /// pac-ret analysis, the expectation is that almost all return instructions
  /// only use register `X30`, and therefore, this vector will probably have
  /// length 1 in the second run.
  std::vector<SmallPtrSet<const MCInst *, 4>> LastInstWritingReg;
  State() {}
  State(unsigned NumRegs, unsigned NumRegsToTrack)
      : NonAutClobRegs(NumRegs), LastInstWritingReg(NumRegsToTrack) {}
  State &operator|=(const State &StateIn) {
    NonAutClobRegs |= StateIn.NonAutClobRegs;
    for (unsigned I = 0; I < LastInstWritingReg.size(); ++I)
      for (const MCInst *J : StateIn.LastInstWritingReg[I])
        LastInstWritingReg[I].insert(J);
    return *this;
  }
  bool operator==(const State &RHS) const {
    return NonAutClobRegs == RHS.NonAutClobRegs &&
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
  OS << "NonAutClobRegs: " << S.NonAutClobRegs << ", ";
  printLastInsts(OS, S.LastInstWritingReg);
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
  OS << "NonAutClobRegs: ";
  RegStatePrinter.print(OS, S.NonAutClobRegs);
  OS << ", ";
  printLastInsts(OS, S.LastInstWritingReg);
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
        RegsToTrackInstsFor(RegsToTrackInstsFor),
        TrackingLastInsts(!RegsToTrackInstsFor.empty()),
        Reg2StateIdx(RegsToTrackInstsFor.empty()
                         ? 0
                         : *llvm::max_element(RegsToTrackInstsFor) + 1,
                     -1) {
    for (unsigned I = 0; I < RegsToTrackInstsFor.size(); ++I)
      Reg2StateIdx[RegsToTrackInstsFor[I]] = I;
  }
  virtual ~PacRetAnalysis() {}

protected:
  const unsigned NumRegs;
  /// RegToTrackInstsFor is the set of registers for which the dataflow analysis
  /// must compute which the last set of instructions writing to it are.
  const std::vector<MCPhysReg> RegsToTrackInstsFor;
  const bool TrackingLastInsts;
  /// Reg2StateIdx maps Register to the index in the vector used in State to
  /// track which instructions last wrote to this register.
  std::vector<uint16_t> Reg2StateIdx;

  SmallPtrSet<const MCInst *, 4> &lastWritingInsts(State &S,
                                                   MCPhysReg Reg) const {
    assert(Reg < Reg2StateIdx.size());
    assert(isTrackingReg(Reg));
    return S.LastInstWritingReg[Reg2StateIdx[Reg]];
  }
  const SmallPtrSet<const MCInst *, 4> &lastWritingInsts(const State &S,
                                                         MCPhysReg Reg) const {
    assert(Reg < Reg2StateIdx.size());
    assert(isTrackingReg(Reg));
    return S.LastInstWritingReg[Reg2StateIdx[Reg]];
  }

  bool isTrackingReg(MCPhysReg Reg) const {
    return llvm::is_contained(RegsToTrackInstsFor, Reg);
  }

  void preflight() {}

  State getStartingStateAtBB(const BinaryBasicBlock &BB) {
    return State(NumRegs, RegsToTrackInstsFor.size());
  }

  State getStartingStateAtPoint(const MCInst &Point) {
    return State(NumRegs, RegsToTrackInstsFor.size());
  }

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

    StateOut |= StateIn;

    LLVM_DEBUG({
      dbgs() << "   merged state: ";
      P.print(dbgs(), StateOut);
      dbgs() << "\n";
    });
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

    State Next = Cur;
    BitVector Written = BitVector(NumRegs, false);
    // Assume a call can clobber all registers, including callee-saved
    // registers. There's a good chance that callee-saved registers will be
    // saved on the stack at some point during execution of the callee.
    // Therefore they should also be considered as potentially modified by an
    // attacker/written to.
    // Also, not all functions may respect the AAPCS ABI rules about
    // caller/callee-saved registers.
    if (BC.MIB->isCall(Point))
      Written.set();
    else
      // FIXME: `getWrittenRegs` only sets the register directly written in the
      // instruction, and the smaller aliasing registers. It does not set the
      // larger aliasing registers. To also set the larger aliasing registers,
      // we'd have to call `getClobberedRegs`.
      // It is unclear if there is any test case which shows a different
      // behaviour between using `getWrittenRegs` vs `getClobberedRegs`. We'd
      // first would like to see such a test case before making a decision
      // on whether using `getClobberedRegs` below would be better.
      // Also see the discussion on this at
      // https://github.com/llvm/llvm-project/pull/122304#discussion_r1939511909
      BC.MIB->getWrittenRegs(Point, Written);
    Next.NonAutClobRegs |= Written;
    // Keep track of this instruction if it writes to any of the registers we
    // need to track that for:
    for (MCPhysReg Reg : RegsToTrackInstsFor)
      if (Written[Reg])
        lastWritingInsts(Next, Reg) = {&Point};

    ErrorOr<MCPhysReg> AutReg = BC.MIB->getAuthenticatedReg(Point);
    if (AutReg && *AutReg != BC.MIB->getNoRegister()) {
      // FIXME: should we use `OnlySmaller=false` below? See similar
      // FIXME about `getWrittenRegs` above and further discussion about this
      // at
      // https://github.com/llvm/llvm-project/pull/122304#discussion_r1939515516
      Next.NonAutClobRegs.reset(
          BC.MIB->getAliases(*AutReg, /*OnlySmaller=*/true));
      if (TrackingLastInsts && isTrackingReg(*AutReg))
        lastWritingInsts(Next, *AutReg).clear();
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
  getLastClobberingInsts(const MCInst Ret, BinaryFunction &BF,
                         const BitVector &UsedDirtyRegs) const {
    if (!TrackingLastInsts)
      return {};
    auto MaybeState = getStateAt(Ret);
    if (!MaybeState)
      llvm_unreachable("Expected State to be present");
    const State &S = *MaybeState;
    // Due to aliasing registers, multiple registers may have been tracked.
    std::set<const MCInst *> LastWritingInsts;
    for (MCPhysReg TrackedReg : UsedDirtyRegs.set_bits()) {
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

FunctionAnalysisResult
Analysis::computeDfState(PacRetAnalysis &PRA, BinaryFunction &BF,
                         MCPlusBuilder::AllocatorIdTy AllocatorId) {
  PRA.run();
  LLVM_DEBUG({
    dbgs() << " After PacRetAnalysis:\n";
    BF.dump();
  });

  FunctionAnalysisResult Result;
  // Now scan the CFG for non-authenticating return instructions that use an
  // overwritten, non-authenticated register as return address.
  BinaryContext &BC = BF.getBinaryContext();
  for (BinaryBasicBlock &BB : BF) {
    for (int64_t I = BB.size() - 1; I >= 0; --I) {
      MCInst &Inst = BB.getInstructionAtIndex(I);
      if (BC.MIB->isReturn(Inst)) {
        ErrorOr<MCPhysReg> MaybeRetReg = BC.MIB->getRegUsedAsRetDest(Inst);
        if (MaybeRetReg.getError()) {
          Result.Diagnostics.push_back(std::make_shared<GenDiag>(
              MCInstInBBReference(&BB, I),
              "Warning: pac-ret analysis could not analyze this return "
              "instruction"));
          continue;
        }
        MCPhysReg RetReg = *MaybeRetReg;
        LLVM_DEBUG({
          dbgs() << "  Found RET inst: ";
          BC.printInstruction(dbgs(), Inst);
          dbgs() << "    RetReg: " << BC.MRI->getName(RetReg)
                 << "; authenticatesReg: "
                 << BC.MIB->isAuthenticationOfReg(Inst, RetReg) << "\n";
        });
        if (BC.MIB->isAuthenticationOfReg(Inst, RetReg))
          break;
        BitVector UsedDirtyRegs = PRA.getStateAt(Inst)->NonAutClobRegs;
        LLVM_DEBUG({
          dbgs() << "  NonAutClobRegs at Ret: ";
          RegStatePrinter RSP(BC);
          RSP.print(dbgs(), UsedDirtyRegs);
          dbgs() << "\n";
        });
        UsedDirtyRegs &= BC.MIB->getAliases(RetReg, /*OnlySmaller=*/true);
        LLVM_DEBUG({
          dbgs() << "  Intersection with RetReg: ";
          RegStatePrinter RSP(BC);
          RSP.print(dbgs(), UsedDirtyRegs);
          dbgs() << "\n";
        });
        if (UsedDirtyRegs.any()) {
          // This return instruction needs to be reported
          Result.Diagnostics.push_back(std::make_shared<Gadget>(
              MCInstInBBReference(&BB, I),
              PRA.getLastClobberingInsts(Inst, BF, UsedDirtyRegs)));
          for (MCPhysReg RetRegWithGadget : UsedDirtyRegs.set_bits())
            Result.RegistersAffected.insert(RetRegWithGadget);
        }
      }
    }
  }
  return Result;
}

void Analysis::runOnFunction(BinaryFunction &BF,
                             MCPlusBuilder::AllocatorIdTy AllocatorId) {
  LLVM_DEBUG({
    dbgs() << "Analyzing in function " << BF.getPrintName() << ", AllocatorId "
           << AllocatorId << "\n";
    BF.dump();
  });

  if (BF.hasCFG()) {
    PacRetAnalysis PRA(BF, AllocatorId, {});
    FunctionAnalysisResult FAR = computeDfState(PRA, BF, AllocatorId);
    if (!FAR.RegistersAffected.empty()) {
      // Redo the analysis, but now also track which instructions last wrote
      // to any of the registers in RetRegsWithGadgets, so that better
      // diagnostics can be produced.
      std::vector<MCPhysReg> RegsToTrack;
      for (MCPhysReg R : FAR.RegistersAffected)
        RegsToTrack.push_back(R);
      PacRetAnalysis PRWIA(BF, AllocatorId, RegsToTrack);
      FAR = computeDfState(PRWIA, BF, AllocatorId);
    }

    // `runOnFunction` is typically getting called from multiple threads in
    // parallel. Therefore, use a lock to avoid data races when storing the
    // result of the analysis in the `AnalysisResults` map.
    {
      std::lock_guard<std::mutex> Lock(AnalysisResultsMutex);
      AnalysisResults[&BF] = FAR;
    }
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
    const MCInstReference RetInst) {
  BinaryBasicBlock *BB = RetInst.getBasicBlock();
  assert(OverwInst.ParentKind == MCInstReference::BasicBlockParent);
  assert(RetInst.ParentKind == MCInstReference::BasicBlockParent);
  MCInstInBBReference OverwInstBB = OverwInst.U.BBRef;
  if (BB == OverwInstBB.BB) {
    // overwriting inst and ret instruction are in the same basic block.
    assert(OverwInstBB.BBIndex < RetInst.U.BBRef.BBIndex);
    OS << "  This happens in the following basic block:\n";
    printBB(BC, BB);
  }
}

void Gadget::generateReport(raw_ostream &OS, const BinaryContext &BC) const {
  GenDiag(RetInst, "non-protected ret found").generateReport(OS, BC);

  BinaryFunction *BF = RetInst.getFunction();
  OS << "  The " << OverwritingRetRegInst.size()
     << " instructions that write to the return register after any "
        "authentication are:\n";
  // Sort by address to ensure output is deterministic.
  std::vector<MCInstReference> ORRI = OverwritingRetRegInst;
  llvm::sort(ORRI, [](const MCInstReference &A, const MCInstReference &B) {
    return A.getAddress() < B.getAddress();
  });
  for (unsigned I = 0; I < ORRI.size(); ++I) {
    MCInstReference InstRef = ORRI[I];
    OS << "  " << (I + 1) << ". ";
    BC.printInstruction(OS, InstRef, InstRef.getAddress(), BF);
  };
  LLVM_DEBUG({
    dbgs() << "  .. OverWritingRetRegInst:\n";
    for (MCInstReference Ref : OverwritingRetRegInst) {
      dbgs() << "    " << Ref << "\n";
    }
  });
  if (OverwritingRetRegInst.size() == 1) {
    const MCInstReference OverwInst = OverwritingRetRegInst[0];
    assert(OverwInst.ParentKind == MCInstReference::BasicBlockParent);
    reportFoundGadgetInSingleBBSingleOverwInst(OS, BC, OverwInst, RetInst);
  }
}

void GenDiag::generateReport(raw_ostream &OS, const BinaryContext &BC) const {
  BinaryFunction *BF = RetInst.getFunction();
  BinaryBasicBlock *BB = RetInst.getBasicBlock();

  OS << "\nGS-PACRET: " << Diag.Text;
  OS << " in function " << BF->getPrintName();
  if (BB)
    OS << ", basic block " << BB->getName();
  OS << ", at address " << llvm::format("%x", RetInst.getAddress()) << "\n";
  OS << "  The return instruction is ";
  BC.printInstruction(OS, RetInst, RetInst.getAddress(), BF);
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
      SkipFunc, "NonPacProtectedRetAnalysis");

  for (BinaryFunction *BF : BC.getAllBinaryFunctions())
    if (AnalysisResults.count(BF) > 0) {
      for (const std::shared_ptr<Annotation> &A :
           AnalysisResults[BF].Diagnostics)
        A->generateReport(outs(), BC);
    }
  return Error::success();
}

} // namespace NonPacProtectedRetAnalysis
} // namespace bolt
} // namespace llvm
