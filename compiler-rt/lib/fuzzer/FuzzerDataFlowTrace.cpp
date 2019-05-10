//===- FuzzerDataFlowTrace.cpp - DataFlowTrace                ---*- C++ -* ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// fuzzer::DataFlowTrace
//===----------------------------------------------------------------------===//

#include "FuzzerDataFlowTrace.h"
#include "FuzzerIO.h"
#include "FuzzerRandom.h"

#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fuzzer {
static const char *kFunctionsTxt = "functions.txt";

bool BlockCoverage::AppendCoverage(const std::string &S) {
  std::stringstream SS(S);
  return AppendCoverage(SS);
}

// Coverage lines have this form:
// CN X Y Z T
// where N is the number of the function, T is the total number of instrumented
// BBs, and X,Y,Z, if present, are the indecies of covered BB.
// BB #0, which is the entry block, is not explicitly listed.
bool BlockCoverage::AppendCoverage(std::istream &IN) {
  std::string L;
  while (std::getline(IN, L, '\n')) {
    if (L.empty() || L[0] != 'C')
      continue; // Ignore non-coverage lines.
    std::stringstream SS(L.c_str() + 1);
    size_t FunctionId  = 0;
    SS >> FunctionId;
    Vector<uint32_t> CoveredBlocks;
    while (true) {
      uint32_t BB = 0;
      SS >> BB;
      if (!SS) break;
      CoveredBlocks.push_back(BB);
    }
    if (CoveredBlocks.empty()) return false;
    uint32_t NumBlocks = CoveredBlocks.back();
    CoveredBlocks.pop_back();
    for (auto BB : CoveredBlocks)
      if (BB >= NumBlocks) return false;
    auto It = Functions.find(FunctionId);
    auto &Counters =
        It == Functions.end()
            ? Functions.insert({FunctionId, Vector<uint32_t>(NumBlocks)})
                  .first->second
            : It->second;

    if (Counters.size() != NumBlocks) return false;  // wrong number of blocks.

    Counters[0]++;
    for (auto BB : CoveredBlocks)
      Counters[BB]++;
  }
  return true;
}

// Assign weights to each function.
// General principles:
//   * any uncovered function gets weight 0.
//   * a function with lots of uncovered blocks gets bigger weight.
//   * a function with a less frequently executed code gets bigger weight.
Vector<double> BlockCoverage::FunctionWeights(size_t NumFunctions) const {
  Vector<double> Res(NumFunctions);
  for (auto It : Functions) {
    auto FunctionID = It.first;
    auto Counters = It.second;
    auto &Weight = Res[FunctionID];
    Weight = 1000.;  // this function is covered.
    Weight /= SmallestNonZeroCounter(Counters);
    Weight *= NumberOfUncoveredBlocks(Counters) + 1;  // make sure it's not 0.
  }
  return Res;
}

void DataFlowTrace::ReadCoverage(const std::string &DirPath) {
  Vector<SizedFile> Files;
  GetSizedFilesFromDir(DirPath, &Files);
  for (auto &SF : Files) {
    auto Name = Basename(SF.File);
    if (Name == kFunctionsTxt) continue;
    std::ifstream IF(SF.File);
    Coverage.AppendCoverage(IF);
  }
}

void DataFlowTrace::Init(const std::string &DirPath,
                         std::string *FocusFunction,
                         Random &Rand) {
  if (DirPath.empty()) return;
  Printf("INFO: DataFlowTrace: reading from '%s'\n", DirPath.c_str());
  Vector<SizedFile> Files;
  GetSizedFilesFromDir(DirPath, &Files);
  std::string L;
  size_t FocusFuncIdx = SIZE_MAX;
  Vector<std::string> FunctionNames;

  // Read functions.txt
  std::ifstream IF(DirPlusFile(DirPath, kFunctionsTxt));
  size_t NumFunctions = 0;
  while (std::getline(IF, L, '\n')) {
    FunctionNames.push_back(L);
    NumFunctions++;
    if (*FocusFunction == L)
      FocusFuncIdx = NumFunctions - 1;
  }

  if (*FocusFunction == "auto") {
    // AUTOFOCUS works like this:
    // * reads the coverage data from the DFT files.
    // * assigns weights to functions based on coverage.
    // * chooses a random function according to the weights.
    ReadCoverage(DirPath);
    auto Weights = Coverage.FunctionWeights(NumFunctions);
    Vector<double> Intervals(NumFunctions + 1);
    std::iota(Intervals.begin(), Intervals.end(), 0);
    auto Distribution = std::piecewise_constant_distribution<double>(
        Intervals.begin(), Intervals.end(), Weights.begin());
    FocusFuncIdx = static_cast<size_t>(Distribution(Rand));
    *FocusFunction = FunctionNames[FocusFuncIdx];
    assert(FocusFuncIdx < NumFunctions);
    Printf("INFO: AUTOFOCUS: %zd %s\n", FocusFuncIdx,
           FunctionNames[FocusFuncIdx].c_str());
    for (size_t i = 0; i < NumFunctions; i++) {
      if (!Weights[i]) continue;
      Printf("  [%zd] W %g\tBB-tot %u\tBB-cov %u\tEntryFreq %u:\t%s\n", i,
             Weights[i], Coverage.GetNumberOfBlocks(i),
             Coverage.GetNumberOfCoveredBlocks(i), Coverage.GetCounter(i, 0),
             FunctionNames[i].c_str());
    }
  }

  if (!NumFunctions || FocusFuncIdx == SIZE_MAX || Files.size() <= 1)
    return;

  // Read traces.
  size_t NumTraceFiles = 0;
  size_t NumTracesWithFocusFunction = 0;
  for (auto &SF : Files) {
    auto Name = Basename(SF.File);
    if (Name == kFunctionsTxt) continue;
    auto ParseError = [&](const char *Err) {
      Printf("DataFlowTrace: parse error: %s\n  File: %s\n  Line: %s\n", Err,
             Name.c_str(), L.c_str());
    };
    NumTraceFiles++;
    // Printf("=== %s\n", Name.c_str());
    std::ifstream IF(SF.File);
    while (std::getline(IF, L, '\n')) {
      if (!L.empty() && L[0] == 'C')
        continue; // Ignore coverage.
      size_t SpacePos = L.find(' ');
      if (SpacePos == std::string::npos)
        return ParseError("no space in the trace line");
      if (L.empty() || L[0] != 'F')
        return ParseError("the trace line doesn't start with 'F'");
      size_t N = std::atol(L.c_str() + 1);
      if (N >= NumFunctions)
        return ParseError("N is greater than the number of functions");
      if (N == FocusFuncIdx) {
        NumTracesWithFocusFunction++;
        const char *Beg = L.c_str() + SpacePos + 1;
        const char *End = L.c_str() + L.size();
        assert(Beg < End);
        size_t Len = End - Beg;
        Vector<uint8_t> V(Len);
        for (size_t I = 0; I < Len; I++) {
          if (Beg[I] != '0' && Beg[I] != '1')
            ParseError("the trace should contain only 0 or 1");
          V[I] = Beg[I] == '1';
        }
        Traces[Name] = V;
        // Print just a few small traces.
        if (NumTracesWithFocusFunction <= 3 && Len <= 16)
          Printf("%s => |%s|\n", Name.c_str(), L.c_str() + SpacePos + 1);
        break;  // No need to parse the following lines.
      }
    }
  }
  assert(NumTraceFiles == Files.size() - 1);
  Printf("INFO: DataFlowTrace: %zd trace files, %zd functions, "
         "%zd traces with focus function\n",
         NumTraceFiles, NumFunctions, NumTracesWithFocusFunction);
}

int CollectDataFlow(const std::string &DFTBinary, const std::string &DirPath,
                    const Vector<SizedFile> &CorporaFiles) {
  Printf("INFO: collecting data flow for %zd files\n", CorporaFiles.size());
  return 0;
}

}  // namespace fuzzer

