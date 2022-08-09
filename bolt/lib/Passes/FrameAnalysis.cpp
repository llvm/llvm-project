//===- bolt/Passes/FrameAnalysis.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FrameAnalysis class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/FrameAnalysis.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Passes/CallGraphWalker.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Timer.h"
#include <fstream>
#include <stack>

#define DEBUG_TYPE "fa"

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltOptCategory;
extern cl::opt<unsigned> Verbosity;

static cl::list<std::string>
    FrameOptFunctionNames("funcs-fop", cl::CommaSeparated,
                          cl::desc("list of functions to apply frame opts"),
                          cl::value_desc("func1,func2,func3,..."));

static cl::opt<std::string> FrameOptFunctionNamesFile(
    "funcs-file-fop",
    cl::desc("file with list of functions to frame optimize"));

static cl::opt<bool> TimeFA("time-fa", cl::desc("time frame analysis steps"),
                            cl::ReallyHidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
    ExperimentalSW("experimental-shrink-wrapping",
                   cl::desc("process functions with stack pointer arithmetic"),
                   cl::ReallyHidden, cl::ZeroOrMore, cl::cat(BoltOptCategory));

bool shouldFrameOptimize(const llvm::bolt::BinaryFunction &Function) {
  if (Function.hasUnknownControlFlow())
    return false;

  if (!FrameOptFunctionNamesFile.empty()) {
    assert(!FrameOptFunctionNamesFile.empty() && "unexpected empty file name");
    std::ifstream FuncsFile(FrameOptFunctionNamesFile, std::ios::in);
    std::string FuncName;
    while (std::getline(FuncsFile, FuncName))
      FrameOptFunctionNames.push_back(FuncName);
    FrameOptFunctionNamesFile = "";
  }

  bool IsValid = true;
  if (!FrameOptFunctionNames.empty()) {
    IsValid = false;
    for (std::string &Name : FrameOptFunctionNames) {
      if (Function.hasName(Name)) {
        IsValid = true;
        break;
      }
    }
  }
  if (!IsValid)
    return false;

  return IsValid;
}
} // namespace opts

namespace llvm {
namespace bolt {

raw_ostream &operator<<(raw_ostream &OS, const FrameIndexEntry &FIE) {
  OS << "FrameIndexEntry<IsLoad: " << FIE.IsLoad << ", IsStore: " << FIE.IsStore
     << ", IsStoreFromReg: " << FIE.IsStoreFromReg
     << ", RegOrImm: " << FIE.RegOrImm << ", StackOffset: ";
  if (FIE.StackOffset < 0)
    OS << "-" << Twine::utohexstr(-FIE.StackOffset);
  else
    OS << "+" << Twine::utohexstr(FIE.StackOffset);
  OS << ", Size: " << static_cast<int>(FIE.Size)
     << ", IsSimple: " << FIE.IsSimple << ">";
  return OS;
}

namespace {

/// This class should be used to iterate through basic blocks in layout order
/// to analyze instructions for frame accesses. The user should call
/// enterNewBB() whenever starting analyzing a new BB and doNext() for each
/// instruction. After doNext(), if isValidAccess() returns true, it means the
/// current instruction accesses the frame and getFIE() may be used to obtain
/// details about this access.
class FrameAccessAnalysis {
  /// We depend on Stack Pointer Tracking to figure out the current SP offset
  /// value at a given program point
  StackPointerTracking &SPT;

  /// Context vars
  const BinaryContext &BC;
  const BinaryFunction &BF;
  // Vars used for storing useful CFI info to give us a hint about how the stack
  // is used in this function
  int SPOffset{0};
  int FPOffset{0};
  int64_t CfaOffset{-8};
  uint16_t CfaReg{7};
  std::stack<std::pair<int64_t, uint16_t>> CFIStack;
  /// Our pointer to access SPT info
  const MCInst *Prev{nullptr};
  /// Info about the last frame access
  bool IsValidAccess{false};
  bool EscapesStackAddress{false};
  FrameIndexEntry FIE;

  bool decodeFrameAccess(const MCInst &Inst) {
    int32_t SrcImm = 0;
    MCPhysReg Reg = 0;
    int64_t StackOffset = 0;
    bool IsIndexed = false;
    if (!BC.MIB->isStackAccess(
            Inst, FIE.IsLoad, FIE.IsStore, FIE.IsStoreFromReg, Reg, SrcImm,
            FIE.StackPtrReg, StackOffset, FIE.Size, FIE.IsSimple, IsIndexed)) {
      return true;
    }

    if (IsIndexed || (!FIE.Size && (FIE.IsLoad || FIE.IsStore))) {
      LLVM_DEBUG(dbgs() << "Giving up on indexed memory access/unknown size\n");
      LLVM_DEBUG(dbgs() << "Blame insn: ");
      LLVM_DEBUG(BC.printInstruction(outs(), Inst, 0, &BF, true, false, false));
      LLVM_DEBUG(Inst.dump());
      return false;
    }

    assert(FIE.Size != 0 || (!FIE.IsLoad && !FIE.IsStore));

    FIE.RegOrImm = SrcImm;
    if (FIE.IsLoad || FIE.IsStoreFromReg)
      FIE.RegOrImm = Reg;

    if (FIE.StackPtrReg == BC.MIB->getStackPointer() && SPOffset != SPT.EMPTY &&
        SPOffset != SPT.SUPERPOSITION) {
      LLVM_DEBUG(
          dbgs() << "Adding access via SP while CFA reg is another one\n");
      FIE.StackOffset = SPOffset + StackOffset;
    } else if (FIE.StackPtrReg == BC.MIB->getFramePointer() &&
               FPOffset != SPT.EMPTY && FPOffset != SPT.SUPERPOSITION) {
      LLVM_DEBUG(
          dbgs() << "Adding access via FP while CFA reg is another one\n");
      FIE.StackOffset = FPOffset + StackOffset;
    } else if (FIE.StackPtrReg ==
               *BC.MRI->getLLVMRegNum(CfaReg, /*isEH=*/false)) {
      FIE.StackOffset = CfaOffset + StackOffset;
    } else {
      LLVM_DEBUG(
          dbgs() << "Found stack access with reg different than cfa reg.\n");
      LLVM_DEBUG(dbgs() << "\tCurrent CFA reg: " << CfaReg
                        << "\n\tStack access reg: " << FIE.StackPtrReg << "\n");
      LLVM_DEBUG(dbgs() << "Blame insn: ");
      LLVM_DEBUG(Inst.dump());
      return false;
    }
    IsValidAccess = true;
    return true;
  }

public:
  FrameAccessAnalysis(BinaryFunction &BF, StackPointerTracking &SPT)
      : SPT(SPT), BC(BF.getBinaryContext()), BF(BF) {}

  void enterNewBB() { Prev = nullptr; }
  const FrameIndexEntry &getFIE() const { return FIE; }
  int getSPOffset() const { return SPOffset; }
  bool isValidAccess() const { return IsValidAccess; }
  bool doesEscapeStackAddress() const { return EscapesStackAddress; }

  bool doNext(const BinaryBasicBlock &BB, const MCInst &Inst) {
    IsValidAccess = false;
    EscapesStackAddress = false;
    std::tie(SPOffset, FPOffset) =
        Prev ? *SPT.getStateAt(*Prev) : *SPT.getStateAt(BB);
    Prev = &Inst;
    // Use CFI information to keep track of which register is being used to
    // access the frame
    if (BC.MIB->isCFI(Inst)) {
      const MCCFIInstruction *CFI = BF.getCFIFor(Inst);
      switch (CFI->getOperation()) {
      case MCCFIInstruction::OpDefCfa:
        CfaOffset = CFI->getOffset();
        [[fallthrough]];
      case MCCFIInstruction::OpDefCfaRegister:
        CfaReg = CFI->getRegister();
        break;
      case MCCFIInstruction::OpDefCfaOffset:
        CfaOffset = CFI->getOffset();
        break;
      case MCCFIInstruction::OpRememberState:
        CFIStack.push(std::make_pair(CfaOffset, CfaReg));
        break;
      case MCCFIInstruction::OpRestoreState: {
        if (CFIStack.empty())
          dbgs() << "Assertion is about to fail: " << BF.getPrintName() << "\n";
        assert(!CFIStack.empty() && "Corrupt CFI stack");
        std::pair<int64_t, uint16_t> &Elem = CFIStack.top();
        CFIStack.pop();
        CfaOffset = Elem.first;
        CfaReg = Elem.second;
        break;
      }
      case MCCFIInstruction::OpAdjustCfaOffset:
        llvm_unreachable("Unhandled AdjustCfaOffset");
        break;
      default:
        break;
      }
      return true;
    }

    if (BC.MIB->escapesVariable(Inst, SPT.HasFramePointer)) {
      EscapesStackAddress = true;
      if (!opts::ExperimentalSW) {
        LLVM_DEBUG(
            dbgs() << "Leaked stack address, giving up on this function.\n");
        LLVM_DEBUG(dbgs() << "Blame insn: ");
        LLVM_DEBUG(Inst.dump());
        return false;
      }
    }

    return decodeFrameAccess(Inst);
  }
};

} // end anonymous namespace

void FrameAnalysis::addArgAccessesFor(MCInst &Inst, ArgAccesses &&AA) {
  if (ErrorOr<ArgAccesses &> OldAA = getArgAccessesFor(Inst)) {
    if (OldAA->AssumeEverything)
      return;
    *OldAA = std::move(AA);
    return;
  }
  if (AA.AssumeEverything) {
    // Index 0 in ArgAccessesVector represents an "assumeeverything" entry
    BC.MIB->addAnnotation(Inst, "ArgAccessEntry", 0U);
    return;
  }
  BC.MIB->addAnnotation(Inst, "ArgAccessEntry",
                        (unsigned)ArgAccessesVector.size());
  ArgAccessesVector.emplace_back(std::move(AA));
}

void FrameAnalysis::addArgInStackAccessFor(MCInst &Inst,
                                           const ArgInStackAccess &Arg) {
  ErrorOr<ArgAccesses &> AA = getArgAccessesFor(Inst);
  if (!AA) {
    addArgAccessesFor(Inst, ArgAccesses(false));
    AA = getArgAccessesFor(Inst);
    assert(AA && "Object setup failed");
  }
  std::set<ArgInStackAccess> &Set = AA->Set;
  assert(!AA->AssumeEverything && "Adding arg to AssumeEverything set");
  Set.emplace(Arg);
}

void FrameAnalysis::addFIEFor(MCInst &Inst, const FrameIndexEntry &FIE) {
  BC.MIB->addAnnotation(Inst, "FrameAccessEntry", (unsigned)FIEVector.size());
  FIEVector.emplace_back(FIE);
}

ErrorOr<ArgAccesses &> FrameAnalysis::getArgAccessesFor(const MCInst &Inst) {
  if (auto Idx = BC.MIB->tryGetAnnotationAs<unsigned>(Inst, "ArgAccessEntry")) {
    assert(ArgAccessesVector.size() > *Idx && "Out of bounds");
    return ArgAccessesVector[*Idx];
  }
  return make_error_code(errc::result_out_of_range);
}

ErrorOr<const ArgAccesses &>
FrameAnalysis::getArgAccessesFor(const MCInst &Inst) const {
  if (auto Idx = BC.MIB->tryGetAnnotationAs<unsigned>(Inst, "ArgAccessEntry")) {
    assert(ArgAccessesVector.size() > *Idx && "Out of bounds");
    return ArgAccessesVector[*Idx];
  }
  return make_error_code(errc::result_out_of_range);
}

ErrorOr<const FrameIndexEntry &>
FrameAnalysis::getFIEFor(const MCInst &Inst) const {
  if (auto Idx =
          BC.MIB->tryGetAnnotationAs<unsigned>(Inst, "FrameAccessEntry")) {
    assert(FIEVector.size() > *Idx && "Out of bounds");
    return FIEVector[*Idx];
  }
  return make_error_code(errc::result_out_of_range);
}

void FrameAnalysis::traverseCG(BinaryFunctionCallGraph &CG) {
  CallGraphWalker CGWalker(CG);

  CGWalker.registerVisitor(
      [&](BinaryFunction *Func) -> bool { return computeArgsAccessed(*Func); });

  CGWalker.walk();

  DEBUG_WITH_TYPE("ra", {
    for (auto &MapEntry : ArgsTouchedMap) {
      const BinaryFunction *Func = MapEntry.first;
      const auto &Set = MapEntry.second;
      dbgs() << "Args accessed for " << Func->getPrintName() << ": ";
      if (!Set.empty() && Set.count(std::make_pair(-1, 0)))
        dbgs() << "assume everything";
      else
        for (const std::pair<int64_t, uint8_t> &Entry : Set)
          dbgs() << "[" << Entry.first << ", " << (int)Entry.second << "] ";
      dbgs() << "\n";
    }
  });
}

bool FrameAnalysis::updateArgsTouchedFor(const BinaryFunction &BF, MCInst &Inst,
                                         int CurOffset) {
  if (!BC.MIB->isCall(Inst))
    return false;

  std::set<int64_t> Res;
  const MCSymbol *TargetSymbol = BC.MIB->getTargetSymbol(Inst);
  // If indirect call, we conservatively assume it accesses all stack positions
  if (TargetSymbol == nullptr) {
    addArgAccessesFor(Inst, ArgAccesses(/*AssumeEverything=*/true));
    if (!FunctionsRequireAlignment.count(&BF)) {
      FunctionsRequireAlignment.insert(&BF);
      return true;
    }
    return false;
  }

  const BinaryFunction *Function = BC.getFunctionForSymbol(TargetSymbol);
  // Call to a function without a BinaryFunction object. Conservatively assume
  // it accesses all stack positions
  if (Function == nullptr) {
    addArgAccessesFor(Inst, ArgAccesses(/*AssumeEverything=*/true));
    if (!FunctionsRequireAlignment.count(&BF)) {
      FunctionsRequireAlignment.insert(&BF);
      return true;
    }
    return false;
  }

  auto Iter = ArgsTouchedMap.find(Function);

  bool Changed = false;
  if (BC.MIB->isTailCall(Inst) && Iter != ArgsTouchedMap.end()) {
    // Ignore checking CurOffset because we can't always reliably determine the
    // offset specially after an epilogue, where tailcalls happen. It should be
    // -8.
    for (std::pair<int64_t, uint8_t> Elem : Iter->second) {
      if (ArgsTouchedMap[&BF].find(Elem) == ArgsTouchedMap[&BF].end()) {
        ArgsTouchedMap[&BF].emplace(Elem);
        Changed = true;
      }
    }
  }
  if (FunctionsRequireAlignment.count(Function) &&
      !FunctionsRequireAlignment.count(&BF)) {
    Changed = true;
    FunctionsRequireAlignment.insert(&BF);
  }
  if (Iter == ArgsTouchedMap.end())
    return Changed;

  if (CurOffset == StackPointerTracking::EMPTY ||
      CurOffset == StackPointerTracking::SUPERPOSITION) {
    addArgAccessesFor(Inst, ArgAccesses(/*AssumeEverything=*/true));
    return Changed;
  }

  for (std::pair<int64_t, uint8_t> Elem : Iter->second) {
    if (Elem.first == -1) {
      addArgAccessesFor(Inst, ArgAccesses(/*AssumeEverything=*/true));
      break;
    }
    LLVM_DEBUG(dbgs() << "Added arg in stack access annotation "
                      << CurOffset + Elem.first << "\n");
    addArgInStackAccessFor(
        Inst, ArgInStackAccess{/*StackOffset=*/CurOffset + Elem.first,
                               /*Size=*/Elem.second});
  }
  return Changed;
}

bool FrameAnalysis::computeArgsAccessed(BinaryFunction &BF) {
  if (!BF.isSimple() || !BF.hasCFG()) {
    LLVM_DEBUG(dbgs() << "Treating " << BF.getPrintName()
                      << " conservatively.\n");
    ArgsTouchedMap[&BF].emplace(std::make_pair(-1, 0));
    if (!FunctionsRequireAlignment.count(&BF)) {
      FunctionsRequireAlignment.insert(&BF);
      return true;
    }
    return false;
  }

  LLVM_DEBUG(dbgs() << "Now computing args accessed for: " << BF.getPrintName()
                    << "\n");
  bool UpdatedArgsTouched = false;
  bool NoInfo = false;
  FrameAccessAnalysis FAA(BF, getSPT(BF));

  for (BinaryBasicBlock *BB : BF.getLayout().blocks()) {
    FAA.enterNewBB();

    for (MCInst &Inst : *BB) {
      if (!FAA.doNext(*BB, Inst) || FAA.doesEscapeStackAddress()) {
        ArgsTouchedMap[&BF].emplace(std::make_pair(-1, 0));
        NoInfo = true;
        break;
      }

      // Check for calls -- attach stack accessing info to them regarding their
      // target
      if (updateArgsTouchedFor(BF, Inst, FAA.getSPOffset()))
        UpdatedArgsTouched = true;

      // Check for stack accesses that affect callers
      if (!FAA.isValidAccess())
        continue;

      const FrameIndexEntry &FIE = FAA.getFIE();
      if (FIE.StackOffset < 0)
        continue;
      if (ArgsTouchedMap[&BF].find(std::make_pair(FIE.StackOffset, FIE.Size)) !=
          ArgsTouchedMap[&BF].end())
        continue;

      // Record accesses to the previous stack frame
      ArgsTouchedMap[&BF].emplace(std::make_pair(FIE.StackOffset, FIE.Size));
      UpdatedArgsTouched = true;
      LLVM_DEBUG({
        dbgs() << "Arg access offset " << FIE.StackOffset << " added to:\n";
        BC.printInstruction(dbgs(), Inst, 0, &BF, true);
      });
    }
    if (NoInfo)
      break;
  }
  if (FunctionsRequireAlignment.count(&BF))
    return UpdatedArgsTouched;

  if (NoInfo) {
    FunctionsRequireAlignment.insert(&BF);
    return true;
  }

  for (BinaryBasicBlock &BB : BF) {
    for (MCInst &Inst : BB) {
      if (BC.MIB->requiresAlignedAddress(Inst)) {
        FunctionsRequireAlignment.insert(&BF);
        return true;
      }
    }
  }
  return UpdatedArgsTouched;
}

bool FrameAnalysis::restoreFrameIndex(BinaryFunction &BF) {
  FrameAccessAnalysis FAA(BF, getSPT(BF));

  LLVM_DEBUG(dbgs() << "Restoring frame indices for \"" << BF.getPrintName()
                    << "\"\n");
  for (BinaryBasicBlock *BB : BF.getLayout().blocks()) {
    LLVM_DEBUG(dbgs() << "\tNow at BB " << BB->getName() << "\n");
    FAA.enterNewBB();

    for (MCInst &Inst : *BB) {
      if (!FAA.doNext(*BB, Inst))
        return false;
      LLVM_DEBUG({
        dbgs() << "\t\tNow at ";
        Inst.dump();
        dbgs() << "\t\t\tSP offset is " << FAA.getSPOffset() << "\n";
      });

      if (FAA.doesEscapeStackAddress()) {
        if (!FunctionsWithStackArithmetic.count(&BF))
          FunctionsWithStackArithmetic.insert(&BF);
        continue;
      }

      if (!FAA.isValidAccess())
        continue;

      const FrameIndexEntry &FIE = FAA.getFIE();

      addFIEFor(Inst, FIE);
      LLVM_DEBUG({
        dbgs() << "Frame index annotation " << FIE << " added to:\n";
        BC.printInstruction(dbgs(), Inst, 0, &BF, true);
      });
    }
  }
  return true;
}

void FrameAnalysis::cleanAnnotations() {
  NamedRegionTimer T("cleanannotations", "clean annotations", "FA",
                     "FA breakdown", opts::TimeFA);

  ParallelUtilities::WorkFuncTy CleanFunction = [&](BinaryFunction &BF) {
    for (BinaryBasicBlock &BB : BF) {
      for (MCInst &Inst : BB) {
        BC.MIB->removeAnnotation(Inst, "ArgAccessEntry");
        BC.MIB->removeAnnotation(Inst, "FrameAccessEntry");
      }
    }
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_LINEAR, CleanFunction,
      ParallelUtilities::PredicateTy(nullptr), "cleanAnnotations");
}

FrameAnalysis::FrameAnalysis(BinaryContext &BC, BinaryFunctionCallGraph &CG)
    : BC(BC) {
  // Position 0 of the vector should be always associated with "assume access
  // everything".
  ArgAccessesVector.emplace_back(ArgAccesses(/*AssumeEverything*/ true));

  if (!opts::NoThreads) {
    NamedRegionTimer T1("precomputespt", "pre-compute spt", "FA",
                        "FA breakdown", opts::TimeFA);
    preComputeSPT();
  }

  {
    NamedRegionTimer T1("traversecg", "traverse call graph", "FA",
                        "FA breakdown", opts::TimeFA);
    traverseCG(CG);
  }

  for (auto &I : BC.getBinaryFunctions()) {
    CountDenominator += I.second.getFunctionScore();

    // "shouldOptimize" for passes that run after finalize
    if (!(I.second.isSimple() && I.second.hasCFG() && !I.second.isIgnored()) ||
        !opts::shouldFrameOptimize(I.second)) {
      ++NumFunctionsNotOptimized;
      continue;
    }

    {
      NamedRegionTimer T1("restorefi", "restore frame index", "FA",
                          "FA breakdown", opts::TimeFA);
      if (!restoreFrameIndex(I.second)) {
        ++NumFunctionsFailedRestoreFI;
        CountFunctionsFailedRestoreFI += I.second.getFunctionScore();
        continue;
      }
    }
    AnalyzedFunctions.insert(&I.second);
  }

  {
    NamedRegionTimer T1("clearspt", "clear spt", "FA", "FA breakdown",
                        opts::TimeFA);
    clearSPTMap();

    // Clean up memory allocated for annotation values
    if (!opts::NoThreads)
      for (MCPlusBuilder::AllocatorIdTy Id : SPTAllocatorsId)
        BC.MIB->freeValuesAllocator(Id);
  }
}

void FrameAnalysis::printStats() {
  outs() << "BOLT-INFO: FRAME ANALYSIS: " << NumFunctionsNotOptimized
         << " function(s) were not optimized.\n"
         << "BOLT-INFO: FRAME ANALYSIS: " << NumFunctionsFailedRestoreFI
         << " function(s) "
         << format("(%.1lf%% dyn cov)",
                   (100.0 * CountFunctionsFailedRestoreFI / CountDenominator))
         << " could not have its frame indices restored.\n";
}

void FrameAnalysis::clearSPTMap() {
  if (opts::NoThreads) {
    SPTMap.clear();
    return;
  }

  ParallelUtilities::WorkFuncTy ClearFunctionSPT = [&](BinaryFunction &BF) {
    std::unique_ptr<StackPointerTracking> &SPTPtr = SPTMap.find(&BF)->second;
    SPTPtr.reset();
  };

  ParallelUtilities::PredicateTy SkipFunc = [&](const BinaryFunction &BF) {
    return !BF.isSimple() || !BF.hasCFG();
  };

  ParallelUtilities::runOnEachFunction(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_LINEAR, ClearFunctionSPT,
      SkipFunc, "clearSPTMap");

  SPTMap.clear();
}

void FrameAnalysis::preComputeSPT() {
  // Make sure that the SPTMap is empty
  assert(SPTMap.size() == 0);

  // Create map entries to allow lock-free parallel execution
  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &BF = BFI.second;
    if (!BF.isSimple() || !BF.hasCFG())
      continue;
    SPTMap.emplace(&BF, std::unique_ptr<StackPointerTracking>());
  }

  // Create an index for the SPT annotation to allow lock-free parallel
  // execution
  BC.MIB->getOrCreateAnnotationIndex("StackPointerTracking");

  // Run SPT in parallel
  ParallelUtilities::WorkFuncWithAllocTy ProcessFunction =
      [&](BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocId) {
        std::unique_ptr<StackPointerTracking> &SPTPtr =
            SPTMap.find(&BF)->second;
        SPTPtr = std::make_unique<StackPointerTracking>(BF, AllocId);
        SPTPtr->run();
      };

  ParallelUtilities::PredicateTy SkipPredicate = [&](const BinaryFunction &BF) {
    return !BF.isSimple() || !BF.hasCFG();
  };

  ParallelUtilities::runOnEachFunctionWithUniqueAllocId(
      BC, ParallelUtilities::SchedulingPolicy::SP_BB_QUADRATIC, ProcessFunction,
      SkipPredicate, "preComputeSPT");
}

} // namespace bolt
} // namespace llvm
