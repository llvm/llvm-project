//===- bolt/Passes/StokeInfo.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Pass to get information for functions for the Stoke Optimization
//  To use the Stoke optimization technique to optimize the HHVM.
//  This Pass solves the two major problems to use the Stoke program without
//  probing its code:
//
//  1. Stoke works on function level, but it is only limited to relative
//  small functions which are loop-free, call-free, exception-free, etc.
//
//  2. Stoke requires much information being manually provided, such as the
//  register usages and memory modification, etc.
//
//  This Pass analyzes all functions and get the required information into
//  .csv file. Next, we use python scripts to process the file, filter
//  out functions for optimization and automatically generate configure files.
//  Finally, these configure files are feed to the Stoke to do the job.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_STOKEINFO_H
#define BOLT_PASSES_STOKEINFO_H

#include "bolt/Passes/BinaryPasses.h"
#include <fstream>

namespace llvm {
namespace bolt {
class DataflowInfoManager;
class RegAnalysis;

/// Structure to hold information needed by Stoke for a function
struct StokeFuncInfo {
  std::string FuncName;
  uint64_t Offset;
  uint64_t Size;
  uint64_t NumInstrs;
  uint64_t NumBlocks;
  bool IsLoopFree;
  unsigned NumLoops;
  unsigned MaxLoopDepth;
  uint64_t HotSize;
  uint64_t TotalSize;
  uint64_t Score;
  bool HasCall;
  std::set<std::string> DefIn;
  std::set<std::string> LiveOut;
  bool HeapOut;
  bool StackOut;
  bool HasRipAddr;
  bool Omitted;

  StokeFuncInfo() { reset(); }

  void reset() {
    FuncName = "";
    Offset = Size = NumInstrs = NumBlocks = 0;
    NumLoops = MaxLoopDepth = 0;
    HotSize = TotalSize = 0;
    Score = 0;
    IsLoopFree = HasCall = HeapOut = StackOut = HasRipAddr = Omitted = false;
    DefIn.clear();
    LiveOut.clear();
  }

  void printCsvHeader(std::ofstream &Outfile) {
    if (Outfile.is_open())
      Outfile << "FuncName,Offset,Size,NumInstrs,NumBlocks,"
              << "IsLoopFree,NumLoops,MaxLoopDepth,"
              << "HotSize,TotalSize,"
              << "Score,"
              << "HasCall,"
              << "DefIn,LiveOut,HeapOut,StackOut,"
              << "HasRipAddr,"
              << "Omitted\n";
  }

  void printData(std::ofstream &Outfile) {
    if (Outfile.is_open()) {
      Outfile << FuncName << "," << Offset << "," << Size << "," << NumInstrs
              << "," << NumBlocks << "," << IsLoopFree << "," << NumLoops << ","
              << MaxLoopDepth << "," << HotSize << "," << TotalSize << ","
              << Score << "," << HasCall << ",\"{ ";
      for (std::string S : DefIn)
        Outfile << "%" << S << " ";
      Outfile << "}\",\"{ ";
      for (std::string S : LiveOut)
        Outfile << "%" << S << " ";
      Outfile << "}\"," << HeapOut << "," << StackOut << "," << HasRipAddr
              << "," << Omitted << "\n";
    }
  }
};

class StokeInfo : public BinaryFunctionPass {

private:
  // stoke --def_in option default value, for X86:
  // rax, rcx, rdx, rsi, rdi, r8, r9, xmm0-xmm7
  BitVector DefaultDefInMask;
  // --live_out option default value: rax, rdx, xmm0, xmm1
  BitVector DefaultLiveOutMask;

  uint16_t NumRegs;

public:
  StokeInfo(const cl::opt<bool> &PrintPass) : BinaryFunctionPass(PrintPass) {}

  const char *getName() const override { return "stoke-get-stat"; }

  void checkInstr(const BinaryFunction &BF, StokeFuncInfo &FuncInfo);

  /// Get all required information for the stoke optimization
  bool checkFunction(BinaryFunction &BF, DataflowInfoManager &DInfo,
                     RegAnalysis &RA, StokeFuncInfo &FuncInfo);

  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
