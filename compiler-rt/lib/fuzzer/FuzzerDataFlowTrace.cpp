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

#include "FuzzerCommand.h"
#include "FuzzerIO.h"
#include "FuzzerRandom.h"
#include "FuzzerSHA1.h"
#include "FuzzerUtil.h"

#include <cstdlib>
#include <fstream>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    assert(FunctionID < NumFunctions);
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

static void DFTStringAppendToVector(Vector<uint8_t> *DFT,
                                    const std::string &DFTString) {
  assert(DFT->size() == DFTString.size());
  for (size_t I = 0, Len = DFT->size(); I < Len; I++)
    (*DFT)[I] = DFTString[I] == '1';
}

// converts a string of '0' and '1' into a Vector<uint8_t>
static Vector<uint8_t> DFTStringToVector(const std::string &DFTString) {
  Vector<uint8_t> DFT(DFTString.size());
  DFTStringAppendToVector(&DFT, DFTString);
  return DFT;
}

static std::ostream &operator<<(std::ostream &OS, const Vector<uint8_t> &DFT) {
  for (auto B : DFT)
    OS << (B ? "1" : "0");
  return OS;
}

static bool ParseError(const char *Err, const std::string &Line) {
  Printf("DataFlowTrace: parse error: %s: Line: %s\n", Err, Line.c_str());
  return false;
};

// TODO(metzman): replace std::string with std::string_view for
// better performance. Need to figure our how to use string_view on Windows.
static bool ParseDFTLine(const std::string &Line, size_t *FunctionNum,
                         std::string *DFTString) {
  if (!Line.empty() && Line[0] != 'F')
    return false; // Ignore coverage.
  size_t SpacePos = Line.find(' ');
  if (SpacePos == std::string::npos)
    return ParseError("no space in the trace line", Line);
  if (Line.empty() || Line[0] != 'F')
    return ParseError("the trace line doesn't start with 'F'", Line);
  *FunctionNum = std::atol(Line.c_str() + 1);
  const char *Beg = Line.c_str() + SpacePos + 1;
  const char *End = Line.c_str() + Line.size();
  assert(Beg < End);
  size_t Len = End - Beg;
  for (size_t I = 0; I < Len; I++) {
    if (Beg[I] != '0' && Beg[I] != '1')
      return ParseError("the trace should contain only 0 or 1", Line);
  }
  *DFTString = Beg;
  return true;
}

bool DataFlowTrace::Init(const std::string &DirPath,
                         std::string *FocusFunction,
                         Random &Rand) {
  if (DirPath.empty()) return false;
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
    return false;

  // Read traces.
  size_t NumTraceFiles = 0;
  size_t NumTracesWithFocusFunction = 0;
  for (auto &SF : Files) {
    auto Name = Basename(SF.File);
    if (Name == kFunctionsTxt) continue;
    NumTraceFiles++;
    // Printf("=== %s\n", Name.c_str());
    std::ifstream IF(SF.File);
    while (std::getline(IF, L, '\n')) {
      size_t FunctionNum = 0;
      std::string DFTString;
      if (ParseDFTLine(L, &FunctionNum, &DFTString) &&
          FunctionNum == FocusFuncIdx) {
        NumTracesWithFocusFunction++;

        if (FunctionNum >= NumFunctions)
          return ParseError("N is greater than the number of functions", L);
        Traces[Name] = DFTStringToVector(DFTString);
        // Print just a few small traces.
        if (NumTracesWithFocusFunction <= 3 && DFTString.size() <= 16)
          Printf("%s => |%s|\n", Name.c_str(), std::string(DFTString).c_str());
        break; // No need to parse the following lines.
      }
    }
  }
  assert(NumTraceFiles == Files.size() - 1);
  Printf("INFO: DataFlowTrace: %zd trace files, %zd functions, "
         "%zd traces with focus function\n",
         NumTraceFiles, NumFunctions, NumTracesWithFocusFunction);
  return true;
}

int CollectDataFlow(const std::string &DFTBinary, const std::string &DirPath,
                    const Vector<SizedFile> &CorporaFiles) {
  Printf("INFO: collecting data flow: bin: %s dir: %s files: %zd\n",
         DFTBinary.c_str(), DirPath.c_str(), CorporaFiles.size());
  MkDir(DirPath);
  auto Temp = TempPath(".dft");
  for (auto &F : CorporaFiles) {
    // For every input F we need to collect the data flow and the coverage.
    // Data flow collection may fail if we request too many DFSan tags at once.
    // So, we start from requesting all tags in range [0,Size) and if that fails
    // we then request tags in [0,Size/2) and [Size/2, Size), and so on.
    // Function number => DFT.
    std::unordered_map<size_t, Vector<uint8_t>> DFTMap;
    std::unordered_set<std::string> Cov;
    std::queue<std::pair<size_t, size_t>> Q;
    Q.push({0, F.Size});
    while (!Q.empty()) {
      auto R = Q.front();
      Printf("\n\n\n********* Trying: [%zd, %zd)\n", R.first, R.second);
      Q.pop();
      Command Cmd;
      Cmd.addArgument(DFTBinary);
      Cmd.addArgument(std::to_string(R.first));
      Cmd.addArgument(std::to_string(R.second));
      Cmd.addArgument(F.File);
      Cmd.addArgument(Temp);
      Printf("CMD: %s\n", Cmd.toString().c_str());
      if (ExecuteCommand(Cmd)) {
        // DFSan has failed, collect tags for two subsets.
        if (R.second - R.first >= 2) {
          size_t Mid = (R.second + R.first) / 2;
          Q.push({R.first, Mid});
          Q.push({Mid, R.second});
        }
      } else {
        Printf("********* Success: [%zd, %zd)\n", R.first, R.second);
        std::ifstream IF(Temp);
        std::string L;
        while (std::getline(IF, L, '\n')) {
          // Data flow collection has succeeded.
          // Merge the results with the other runs.
          if (L.empty()) continue;
          if (L[0] == 'C') {
            // Take coverage lines as is, they will be the same in all attempts.
            Cov.insert(L);
          } else if (L[0] == 'F') {
            size_t FunctionNum = 0;
            std::string DFTString;
            if (ParseDFTLine(L, &FunctionNum, &DFTString)) {
              auto &DFT = DFTMap[FunctionNum];
              if (DFT.empty()) {
                // Haven't seen this function before, take DFT as is.
                DFT = DFTStringToVector(DFTString);
              } else if (DFT.size() == DFTString.size()) {
                // Have seen this function already, merge DFTs.
                DFTStringAppendToVector(&DFT, DFTString);
              }
            }
          }
        }
      }
    }
    auto OutPath = DirPlusFile(DirPath, Hash(FileToVector(F.File)));
    // Dump combined DFT to disk.
    Printf("Producing DFT for %s\n", OutPath.c_str());
    std::ofstream OF(OutPath);
    for (auto &DFT: DFTMap)
      OF << "F" << DFT.first << " " << DFT.second << std::endl;
    for (auto &C : Cov)
      OF << C << std::endl;
  }
  RemoveFile(Temp);
  // Write functions.txt.
  Command Cmd;
  Cmd.addArgument(DFTBinary);
  Cmd.setOutputFile(DirPlusFile(DirPath, "functions.txt"));
  ExecuteCommand(Cmd);
  return 0;
}

}  // namespace fuzzer
