//===- bolt/Passes/StokeInfo.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the StokeInfo class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/StokeInfo.h"
#include "bolt/Passes/BinaryFunctionCallGraph.h"
#include "bolt/Passes/DataflowInfoManager.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "stoke"

using namespace llvm;
using namespace bolt;

namespace opts {
cl::OptionCategory StokeOptCategory("STOKE pass options");

static cl::opt<std::string>
StokeOutputDataFilename("stoke-out",
  cl::desc("output data (.csv) for Stoke's use"),
  cl::Optional,
  cl::cat(StokeOptCategory));
}

namespace llvm {
namespace bolt {

void getRegNameFromBitVec(const BinaryContext &BC, const BitVector &RegV,
                          std::set<std::string> *NameVec = nullptr) {
  for (int RegIdx : RegV.set_bits()) {
    LLVM_DEBUG(dbgs() << BC.MRI->getName(RegIdx) << " ");
    if (NameVec)
      NameVec->insert(std::string(BC.MRI->getName(RegIdx)));
  }
  LLVM_DEBUG(dbgs() << "\n");
}

void StokeInfo::checkInstr(const BinaryFunction &BF, StokeFuncInfo &FuncInfo) {
  MCPlusBuilder *MIB = BF.getBinaryContext().MIB.get();
  BitVector RegV(NumRegs, false);
  for (const BinaryBasicBlock *BB : BF.getLayout().blocks()) {
    if (BB->empty())
      continue;

    // Skip function with exception handling.
    if (BB->throw_size() || BB->lp_size()) {
      FuncInfo.Omitted = true;
      return;
    }

    for (const MCInst &It : *BB) {
      if (MIB->isPseudo(It))
        continue;
      // skip function with exception handling yet
      if (MIB->isInvoke(It)) {
        FuncInfo.Omitted = true;
        return;
      }
      // check if this function contains call instruction
      if (MIB->isCall(It)) {
        FuncInfo.HasCall = true;
        const MCSymbol *TargetSymbol = MIB->getTargetSymbol(It);
        // if it is an indirect call, skip
        if (TargetSymbol == nullptr) {
          FuncInfo.Omitted = true;
          return;
        }
      }
      // check if this function modify stack or heap
      // TODO: more accurate analysis
      bool IsPush = MIB->isPush(It);
      bool IsRipAddr = MIB->hasPCRelOperand(It);
      if (IsPush)
        FuncInfo.StackOut = true;

      if (MIB->mayStore(It) && !IsPush && !IsRipAddr)
        FuncInfo.HeapOut = true;

      if (IsRipAddr)
        FuncInfo.HasRipAddr = true;
    } // end of for (auto &It : ...)
  }   // end of for (auto *BB : ...)
}

bool StokeInfo::checkFunction(BinaryFunction &BF, DataflowInfoManager &DInfo,
                              RegAnalysis &RA, StokeFuncInfo &FuncInfo) {

  std::string Name = BF.getSymbol()->getName().str();

  if (!BF.isSimple() || BF.isMultiEntry() || BF.empty())
    return false;
  BF.getBinaryContext().outs()
      << " STOKE-INFO: analyzing function " << Name << "\n";

  FuncInfo.FuncName = Name;
  FuncInfo.Offset = BF.getFileOffset();
  FuncInfo.Size = BF.getMaxSize();
  FuncInfo.NumInstrs = BF.getNumNonPseudos();
  FuncInfo.NumBlocks = BF.size();
  // early stop for large functions
  if (FuncInfo.NumInstrs > 500)
    return false;

  FuncInfo.IsLoopFree = BF.isLoopFree();
  if (!FuncInfo.IsLoopFree) {
    const BinaryLoopInfo &BLI = BF.getLoopInfo();
    FuncInfo.NumLoops = BLI.OuterLoops;
    FuncInfo.MaxLoopDepth = BLI.MaximumDepth;
  }

  FuncInfo.HotSize = BF.estimateHotSize();
  FuncInfo.TotalSize = BF.estimateSize();
  FuncInfo.Score = BF.getFunctionScore();

  checkInstr(BF, FuncInfo);

  // register analysis
  BinaryBasicBlock &EntryBB = BF.front();
  assert(EntryBB.isEntryPoint() && "Weird, this should be the entry block!");

  MCInst *FirstNonPseudo = EntryBB.getFirstNonPseudoInstr();
  if (!FirstNonPseudo)
    return false;

  LLVM_DEBUG(dbgs() << "\t [DefIn]\n\t ");
  BitVector LiveInBV =
      *(DInfo.getLivenessAnalysis().getStateAt(FirstNonPseudo));
  LiveInBV &= DefaultDefInMask;
  getRegNameFromBitVec(BF.getBinaryContext(), LiveInBV, &FuncInfo.DefIn);

  LLVM_DEBUG(dbgs() << "\t [LiveOut]\n\t ");
  BitVector LiveOutBV = RA.getFunctionClobberList(&BF);
  LiveOutBV &= DefaultLiveOutMask;
  getRegNameFromBitVec(BF.getBinaryContext(), LiveOutBV, &FuncInfo.LiveOut);

  BF.getBinaryContext().outs() << " STOKE-INFO: end function \n";
  return true;
}

Error StokeInfo::runOnFunctions(BinaryContext &BC) {
  BC.outs() << "STOKE-INFO: begin of stoke pass\n";

  std::ofstream Outfile;
  if (!opts::StokeOutputDataFilename.empty()) {
    Outfile.open(opts::StokeOutputDataFilename);
  } else {
    BC.errs() << "STOKE-INFO: output file is required\n";
    return Error::success();
  }

  // check some context meta data
  LLVM_DEBUG(dbgs() << "\tTarget: " << BC.TheTarget->getName() << "\n");
  LLVM_DEBUG(dbgs() << "\tTripleName " << BC.TripleName << "\n");
  LLVM_DEBUG(dbgs() << "\tgetNumRegs " << BC.MRI->getNumRegs() << "\n");

  BinaryFunctionCallGraph CG = buildCallGraph(BC);
  RegAnalysis RA(BC, &BC.getBinaryFunctions(), &CG);

  NumRegs = BC.MRI->getNumRegs();
  assert(NumRegs > 0 && "STOKE-INFO: the target register number is incorrect!");

  DefaultDefInMask.resize(NumRegs, false);
  DefaultLiveOutMask.resize(NumRegs, false);

  BC.MIB->getDefaultDefIn(DefaultDefInMask);
  BC.MIB->getDefaultLiveOut(DefaultLiveOutMask);

  getRegNameFromBitVec(BC, DefaultDefInMask);
  getRegNameFromBitVec(BC, DefaultLiveOutMask);

  StokeFuncInfo FuncInfo;
  // analyze all functions
  FuncInfo.printCsvHeader(Outfile);
  for (auto &BF : BC.getBinaryFunctions()) {
    DataflowInfoManager DInfo(BF.second, &RA, nullptr);
    FuncInfo.reset();
    if (checkFunction(BF.second, DInfo, RA, FuncInfo))
      FuncInfo.printData(Outfile);
  }

  BC.outs() << "STOKE-INFO: end of stoke pass\n";
  return Error::success();
}

} // namespace bolt
} // namespace llvm
