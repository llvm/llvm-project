//===- bolt/Passes/ContinuityStats.cpp - function cfg continuity analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Conduct function CFG continuity analysis.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/ContinuityStats.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/Support/CommandLine.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>

#define DEBUG_TYPE "bolt-opts"

using namespace llvm;
using namespace bolt;

namespace opts {
extern cl::opt<unsigned> Verbosity;
cl::opt<unsigned> NumTopFunctions(
    "num-top-functions",
    cl::desc(
        "number of hottest functions to print aggregated CFG discontinuity stats of."),
    cl::init(1000), cl::ZeroOrMore, cl::Hidden, cl::cat(BoltOptCategory));
cl::opt<bool> PrintBucketedStats(
    "print-bucketed-stats",
    cl::desc("print CFG discontinuity stats for the top functions divided into buckets "
             "based on their execution counts."),
    cl::Hidden, cl::cat(BoltCategory));
cl::opt<unsigned>
    NumFunctionsPerBucket("num-functions-per-bucket",
                          cl::desc("maximum number of functions per bucket."),
                          cl::init(500), cl::ZeroOrMore, cl::Hidden,
                          cl::cat(BoltOptCategory));
cl::opt<unsigned>
    MinNumFunctions("min-num-functions",
                          cl::desc("minimum number of hot functions in the binary to "
                          "trigger profile CFG continuity check."),
                          cl::init(5), cl::ZeroOrMore, cl::Hidden,
                          cl::cat(BoltOptCategory));
} // namespace opts

namespace {
using FunctionListType = std::vector<const BinaryFunction *>;
using function_iterator = FunctionListType::iterator;

template <typename T>
void printDistribution(raw_ostream &OS, std::vector<T> &values,
                       bool Fraction = false) {
  if (values.empty())
    return;
  // Sort values from largest to smallest and print the MAX, TOP 1%, 5%, 10%,
  // 20%, 50%, 80%, MIN. If Fraction is true, then values are printed as
  // fractions instead of integers.
  std::sort(values.begin(), values.end());

  auto printLine = [&](std::string Text, double Percent) {
    int Rank = int(values.size() * (1.0 - Percent / 100));
    if (Percent == 0)
      Rank = values.size() - 1;
    if (Fraction)
      OS << "  " << Text << std::string(9 - Text.length(), ' ') << ": "
         << format("%.2lf%%", values[Rank] * 100) << "\n";
    else
      OS << "  " << Text << std::string(9 - Text.length(), ' ') << ": "
         << values[Rank] << "\n";
  };

  printLine("MAX", 0);
  int percentages[] = {1, 5, 10, 20, 50, 80};
  for (size_t i = 0; i < sizeof(percentages) / sizeof(percentages[0]); ++i) {
    printLine("TOP " + std::to_string(percentages[i]) + "%", percentages[i]);
  }
  printLine("MIN", 100);
}

void printCFGContinuityStats(raw_ostream &OS,
                               iterator_range<function_iterator> &Functions, 
                               bool Verbose=false) {
  // Given a perfect profile, every positive-execution-count BB should be
  // connected to an entry of the function through a positive-execution-count
  // directed path in the control flow graph.
  std::vector<size_t> NumUnreachables;
  std::vector<size_t> SumECUnreachables;
  std::vector<double> FractionECUnreachables;

  for (auto it = Functions.begin(); it != Functions.end(); ++it) {
    const BinaryFunction *Function = *it;
    if (Function->size() <= 1)
      continue;

    // Compute the sum of all BB execution counts (ECs).
    size_t NumPosECBBs = 0;
    size_t SumAllBBEC = 0;
    for (const BinaryBasicBlock &BB : *Function) {
      size_t BBEC = BB.getKnownExecutionCount();
      NumPosECBBs += BBEC > 0 ? 1 : 0;
      SumAllBBEC += BBEC;
    }

    // Perform BFS on subgraph of CFG induced by positive weight edges.
    // Compute the number of BBs reachable from the entry(s) of the function and
    // the sum of their execution counts (ECs).
    std::unordered_map<unsigned, const BinaryBasicBlock *> IndexToBB;
    std::unordered_set<unsigned> Visited;
    std::queue<unsigned> Queue;
    for (const BinaryBasicBlock &BB : *Function) {
      // Make sure BB.getIndex() is not already in IndexToBB.
      assert(IndexToBB.find(BB.getIndex()) == IndexToBB.end());
      IndexToBB[BB.getIndex()] = &BB;
      if (BB.isEntryPoint() && BB.getKnownExecutionCount() > 0) {
        Queue.push(BB.getIndex());
        Visited.insert(BB.getIndex());
      }
    }
    while (!Queue.empty()) {
      unsigned BBIndex = Queue.front();
      const BinaryBasicBlock *BB = IndexToBB[BBIndex];
      Queue.pop();
      auto SuccBIIter = BB->branch_info_begin();
      for (BinaryBasicBlock *Succ : BB->successors()) {
        uint64_t Count = SuccBIIter->Count;
        if (Count == BinaryBasicBlock::COUNT_NO_PROFILE || Count == 0) {
          ++SuccBIIter;
          continue;
        }
        if (!Visited.insert(Succ->getIndex()).second) {
          ++SuccBIIter;
          continue;
        }
        Queue.push(Succ->getIndex());
        ++SuccBIIter;
      }
    }

    size_t NumReachableBBs = Visited.size();

    // Loop through Visited, and sum the corresponding BBs' execution counts (ECs).
    size_t SumReachableBBEC = 0;
    for (unsigned BBIndex : Visited) {
      const BinaryBasicBlock *BB = IndexToBB[BBIndex];
      SumReachableBBEC += BB->getKnownExecutionCount();
    }

    size_t NumPosECBBsUnreachableFromEntry = NumPosECBBs - NumReachableBBs;
    size_t SumUnreachableBBEC = SumAllBBEC - SumReachableBBEC;
    double FractionECUnreachable = (double)SumUnreachableBBEC / SumAllBBEC;

    if (opts::Verbosity >= 2 && FractionECUnreachable > 0.1 &&
        SumUnreachableBBEC > 50) {
      OS << "Non-trivial CFG discontinuity observed in function "
         << Function->getPrintName() << "\n";
      LLVM_DEBUG(Function->dump());
    }

    NumUnreachables.push_back(NumPosECBBsUnreachableFromEntry);
    SumECUnreachables.push_back(SumUnreachableBBEC);
    FractionECUnreachables.push_back(FractionECUnreachable);
  }

  if (!Verbose) {
    if (FractionECUnreachables.empty()) {
      OS << "no functions have more than 1 basic block and hence no CFG discontinuity.\n";
      return;
    }
    std::sort(FractionECUnreachables.begin(), FractionECUnreachables.end());
    int Rank = int(FractionECUnreachables.size() * 0.95);
    OS << format("the TOP 5%% function CFG discontinuity is %.2lf%%", FractionECUnreachables[Rank] * 100) << "\n";
    return;
  }
  
  OS << format("Focus on %zu (%.2lf%%) of considered functions that have at "
               "least 2 basic blocks\n",
               SumECUnreachables.size(),
               100.0 * (double)SumECUnreachables.size() /
                   (std::distance(Functions.begin(), Functions.end())));

  if (!NumUnreachables.empty()) {
    OS << "- Distribution of NUM(unreachable POS BBs) among all focal "
          "functions\n";
    printDistribution(OS, NumUnreachables);
  }
  if (!SumECUnreachables.empty()) {
    OS << "- Distribution of SUM(unreachable POS BBs) among all focal "
          "functions\n";
    printDistribution(OS, SumECUnreachables);
  }
  if (!FractionECUnreachables.empty()) {
    OS << "- Distribution of [(SUM(unreachable POS BBs) / SUM(all "
          "POS BBs))] among all focal functions\n";
    printDistribution(OS, FractionECUnreachables, /*Fraction=*/true);
  }
}

void printAll(BinaryContext &BC,
              size_t NumFunctionsPerBucket,
              size_t NumTopFunctions) {

  // Create a list of functions with valid profiles.
  FunctionListType ValidFunctions;
  for (const auto &BFI : BC.getBinaryFunctions()) {
    const BinaryFunction *Function = &BFI.second;
    if (Function->empty() || !Function->hasValidProfile() ||
        !Function->isSimple())
      continue;
    ValidFunctions.push_back(Function);
  }

  // Sort the list of functions by execution counts (reverse).
  llvm::sort(ValidFunctions,
             [&](const BinaryFunction *A, const BinaryFunction *B) {
               return A->getKnownExecutionCount() > B->getKnownExecutionCount();
             });

  size_t RealNumTopFunctions = std::min(NumTopFunctions, ValidFunctions.size());
  if (RealNumTopFunctions <= opts::MinNumFunctions)
    return;
  BC.outs() << format("BOLT-INFO: among the hottest %zu functions ", RealNumTopFunctions);
  iterator_range<function_iterator> Functions(
      ValidFunctions.begin(), ValidFunctions.begin() + RealNumTopFunctions);
  printCFGContinuityStats(BC.outs(), Functions, /*Verbose=*/false);

  // Print more detailed bucketed stats if requested.
  if (opts::PrintBucketedStats) {
    size_t PerBucketSize = std::min(NumFunctionsPerBucket, ValidFunctions.size());
    if (PerBucketSize == 0)
      return;
    size_t NumBuckets = RealNumTopFunctions / PerBucketSize +
                        (RealNumTopFunctions % PerBucketSize != 0);
    BC.outs() << format("Detailed stats for %zu buckets, each with at most %zu functions:\n",
                NumBuckets, PerBucketSize);
    BC.outs() << "For each considered function, identify positive execution-count basic blocks\n"
       << "(abbr. POS BBs) that are *unreachable* from the function entry through a\n"
       << "positive-execution-count path.\n";

    // For each bucket, print the CFG continuity stats of the functions in the bucket.
    for (size_t BucketIndex = 0; BucketIndex < NumBuckets; ++BucketIndex) {
      const size_t StartIndex = BucketIndex * PerBucketSize;
      size_t EndIndex = std::min(StartIndex + PerBucketSize, NumTopFunctions);
      EndIndex = std::min(EndIndex, ValidFunctions.size());
      iterator_range<function_iterator> Functions(
          ValidFunctions.begin() + StartIndex, ValidFunctions.begin() + EndIndex);
      const size_t MaxFunctionExecutionCount =
          ValidFunctions[StartIndex]->getKnownExecutionCount();
      const size_t MinFunctionExecutionCount =
          ValidFunctions[EndIndex - 1]->getKnownExecutionCount();
      BC.outs() << format("----------------\n|   Bucket %zu:  "
                  "|\n----------------\nExecution counts of the %zu functions in the bucket: "
                  "%zu-%zu\n",
                  BucketIndex + 1, EndIndex - StartIndex,
                  MinFunctionExecutionCount, MaxFunctionExecutionCount);
      printCFGContinuityStats(BC.outs(), Functions, /*Verbose=*/true);
    }
  }
}
} // namespace

Error PrintContinuityStats::runOnFunctions(BinaryContext &BC) {
  printAll(BC, opts::NumFunctionsPerBucket, opts::NumTopFunctions);
  return Error::success();
}
