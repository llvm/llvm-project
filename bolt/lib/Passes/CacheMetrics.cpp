//===- bolt/Passes/CacheMetrics.cpp - Metrics for instruction cache -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CacheMetrics class and functions for showing metrics
// of cache lines.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/CacheMetrics.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/Support/CommandLine.h"
#include <unordered_map>

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

extern cl::opt<double> ForwardWeight;
extern cl::opt<double> BackwardWeight;
extern cl::opt<unsigned> ForwardDistance;
extern cl::opt<unsigned> BackwardDistance;
extern cl::opt<unsigned> ITLBPageSize;
extern cl::opt<unsigned> ITLBEntries;

} // namespace opts

namespace {

/// Initialize and return a position map for binary basic blocks
void extractBasicBlockInfo(
    const std::vector<BinaryFunction *> &BinaryFunctions,
    std::unordered_map<BinaryBasicBlock *, uint64_t> &BBAddr,
    std::unordered_map<BinaryBasicBlock *, uint64_t> &BBSize) {

  for (BinaryFunction *BF : BinaryFunctions) {
    const BinaryContext &BC = BF->getBinaryContext();
    for (BinaryBasicBlock &BB : *BF) {
      if (BF->isSimple() || BC.HasRelocations) {
        // Use addresses/sizes as in the output binary
        BBAddr[&BB] = BB.getOutputAddressRange().first;
        BBSize[&BB] = BB.getOutputSize();
      } else {
        // Output ranges should match the input if the body hasn't changed
        BBAddr[&BB] = BB.getInputAddressRange().first + BF->getAddress();
        BBSize[&BB] = BB.getOriginalSize();
      }
    }
  }
}

/// Calculate TSP metric, which quantifies the number of fallthrough jumps in
/// the ordering of basic blocks
double
calcTSPScore(const std::vector<BinaryFunction *> &BinaryFunctions,
             const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBAddr,
             const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBSize) {

  double Score = 0;
  for (BinaryFunction *BF : BinaryFunctions) {
    if (!BF->hasProfile())
      continue;
    for (BinaryBasicBlock &SrcBB : *BF) {
      auto BI = SrcBB.branch_info_begin();
      for (BinaryBasicBlock *DstBB : SrcBB.successors()) {
        if (&SrcBB != DstBB &&
            BI->Count != BinaryBasicBlock::COUNT_NO_PROFILE &&
            BBAddr.at(&SrcBB) + BBSize.at(&SrcBB) == BBAddr.at(DstBB))
          Score += BI->Count;
        ++BI;
      }
    }
  }
  return Score;
}

/// Calculate Ext-TSP metric, which quantifies the expected number of i-cache
/// misses for a given ordering of basic blocks
double calcExtTSPScore(
    const std::vector<BinaryFunction *> &BinaryFunctions,
    const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBAddr,
    const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBSize) {

  double Score = 0.0;
  for (BinaryFunction *BF : BinaryFunctions) {
    if (!BF->hasProfile())
      continue;
    for (BinaryBasicBlock &SrcBB : *BF) {
      auto BI = SrcBB.branch_info_begin();
      for (BinaryBasicBlock *DstBB : SrcBB.successors()) {
        if (DstBB != &SrcBB)
          Score +=
              CacheMetrics::extTSPScore(BBAddr.at(&SrcBB), BBSize.at(&SrcBB),
                                        BBAddr.at(DstBB), BI->Count);
        ++BI;
      }
    }
  }
  return Score;
}

using Predecessors = std::vector<std::pair<BinaryFunction *, uint64_t>>;

/// Build a simplified version of the call graph: For every function, keep
/// its callers and the frequencies of the calls
std::unordered_map<const BinaryFunction *, Predecessors>
extractFunctionCalls(const std::vector<BinaryFunction *> &BinaryFunctions) {
  std::unordered_map<const BinaryFunction *, Predecessors> Calls;

  for (BinaryFunction *SrcFunction : BinaryFunctions) {
    const BinaryContext &BC = SrcFunction->getBinaryContext();
    for (BinaryBasicBlock *BB : SrcFunction->getLayout().blocks()) {
      // Find call instructions and extract target symbols from each one
      for (MCInst &Inst : *BB) {
        if (!BC.MIB->isCall(Inst))
          continue;

        // Call info
        const MCSymbol *DstSym = BC.MIB->getTargetSymbol(Inst);
        uint64_t Count = BB->getKnownExecutionCount();
        // Ignore calls w/o information
        if (DstSym == nullptr || Count == 0)
          continue;

        const BinaryFunction *DstFunction = BC.getFunctionForSymbol(DstSym);
        // Ignore recursive calls
        if (DstFunction == nullptr || DstFunction->getLayout().block_empty() ||
            DstFunction == SrcFunction)
          continue;

        // Record the call
        Calls[DstFunction].emplace_back(SrcFunction, Count);
      }
    }
  }
  return Calls;
}

/// Compute expected hit ratio of the i-TLB cache (optimized by HFSortPlus alg).
/// Given an assignment of functions to the i-TLB pages), we divide all
/// functions calls into two categories:
/// - 'short' ones that have a caller-callee distance less than a page;
/// - 'long' ones where the distance exceeds a page.
/// The short calls are likely to result in a i-TLB cache hit. For the long
/// ones, the hit/miss result depends on the 'hotness' of the page (i.e., how
/// often the page is accessed). Assuming that functions are sent to the i-TLB
/// cache in a random order, the probability that a page is present in the cache
/// is proportional to the number of samples corresponding to the functions on
/// the page. The following procedure detects short and long calls, and
/// estimates the expected number of cache misses for the long ones.
double expectedCacheHitRatio(
    const std::vector<BinaryFunction *> &BinaryFunctions,
    const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBAddr,
    const std::unordered_map<BinaryBasicBlock *, uint64_t> &BBSize) {

  const double PageSize = opts::ITLBPageSize;
  const uint64_t CacheEntries = opts::ITLBEntries;
  std::unordered_map<const BinaryFunction *, Predecessors> Calls =
      extractFunctionCalls(BinaryFunctions);
  // Compute 'hotness' of the functions
  double TotalSamples = 0;
  std::unordered_map<BinaryFunction *, double> FunctionSamples;
  for (BinaryFunction *BF : BinaryFunctions) {
    double Samples = 0;
    for (std::pair<BinaryFunction *, uint64_t> Pair : Calls[BF])
      Samples += Pair.second;
    Samples = std::max(Samples, (double)BF->getKnownExecutionCount());
    FunctionSamples[BF] = Samples;
    TotalSamples += Samples;
  }

  // Compute 'hotness' of the pages
  std::unordered_map<uint64_t, double> PageSamples;
  for (BinaryFunction *BF : BinaryFunctions) {
    if (BF->getLayout().block_empty())
      continue;
    double Page = BBAddr.at(BF->getLayout().block_front()) / PageSize;
    PageSamples[Page] += FunctionSamples.at(BF);
  }

  // Computing the expected number of misses for every function
  double Misses = 0;
  for (BinaryFunction *BF : BinaryFunctions) {
    // Skip the function if it has no samples
    if (BF->getLayout().block_empty() || FunctionSamples.at(BF) == 0.0)
      continue;
    double Samples = FunctionSamples.at(BF);
    double Page = BBAddr.at(BF->getLayout().block_front()) / PageSize;
    // The probability that the page is not present in the cache
    double MissProb = pow(1.0 - PageSamples[Page] / TotalSamples, CacheEntries);

    // Processing all callers of the function
    for (std::pair<BinaryFunction *, uint64_t> Pair : Calls[BF]) {
      BinaryFunction *SrcFunction = Pair.first;
      double SrcPage =
          BBAddr.at(SrcFunction->getLayout().block_front()) / PageSize;
      // Is this a 'long' or a 'short' call?
      if (Page != SrcPage) {
        // This is a miss
        Misses += MissProb * Pair.second;
      }
      Samples -= Pair.second;
    }
    assert(Samples >= 0.0 && "Function samples computed incorrectly");
    // The remaining samples likely come from the jitted code
    Misses += Samples * MissProb;
  }

  return 100.0 * (1.0 - Misses / TotalSamples);
}

} // namespace

double CacheMetrics::extTSPScore(uint64_t SrcAddr, uint64_t SrcSize,
                                 uint64_t DstAddr, uint64_t Count) {
  assert(Count != BinaryBasicBlock::COUNT_NO_PROFILE);

  // Fallthrough
  if (SrcAddr + SrcSize == DstAddr) {
    // Assume that FallthroughWeight = 1.0 after normalization
    return static_cast<double>(Count);
  }
  // Forward
  if (SrcAddr + SrcSize < DstAddr) {
    const uint64_t Dist = DstAddr - (SrcAddr + SrcSize);
    if (Dist <= opts::ForwardDistance) {
      double Prob = 1.0 - static_cast<double>(Dist) / opts::ForwardDistance;
      return opts::ForwardWeight * Prob * Count;
    }
    return 0;
  }
  // Backward
  const uint64_t Dist = SrcAddr + SrcSize - DstAddr;
  if (Dist <= opts::BackwardDistance) {
    double Prob = 1.0 - static_cast<double>(Dist) / opts::BackwardDistance;
    return opts::BackwardWeight * Prob * Count;
  }
  return 0;
}

void CacheMetrics::printAll(const std::vector<BinaryFunction *> &BFs) {
  // Stats related to hot-cold code splitting
  size_t NumFunctions = 0;
  size_t NumProfiledFunctions = 0;
  size_t NumHotFunctions = 0;
  size_t NumBlocks = 0;
  size_t NumHotBlocks = 0;

  size_t TotalCodeMinAddr = std::numeric_limits<size_t>::max();
  size_t TotalCodeMaxAddr = 0;
  size_t HotCodeMinAddr = std::numeric_limits<size_t>::max();
  size_t HotCodeMaxAddr = 0;

  for (BinaryFunction *BF : BFs) {
    NumFunctions++;
    if (BF->hasProfile())
      NumProfiledFunctions++;
    if (BF->hasValidIndex())
      NumHotFunctions++;
    for (const BinaryBasicBlock &BB : *BF) {
      NumBlocks++;
      size_t BBAddrMin = BB.getOutputAddressRange().first;
      size_t BBAddrMax = BB.getOutputAddressRange().second;
      TotalCodeMinAddr = std::min(TotalCodeMinAddr, BBAddrMin);
      TotalCodeMaxAddr = std::max(TotalCodeMaxAddr, BBAddrMax);
      if (BF->hasValidIndex() && !BB.isCold()) {
        NumHotBlocks++;
        HotCodeMinAddr = std::min(HotCodeMinAddr, BBAddrMin);
        HotCodeMaxAddr = std::max(HotCodeMaxAddr, BBAddrMax);
      }
    }
  }

  outs() << format("  There are %zu functions;", NumFunctions)
         << format(" %zu (%.2lf%%) are in the hot section,", NumHotFunctions,
                   100.0 * NumHotFunctions / NumFunctions)
         << format(" %zu (%.2lf%%) have profile\n", NumProfiledFunctions,
                   100.0 * NumProfiledFunctions / NumFunctions);
  outs() << format("  There are %zu basic blocks;", NumBlocks)
         << format(" %zu (%.2lf%%) are in the hot section\n", NumHotBlocks,
                   100.0 * NumHotBlocks / NumBlocks);

  assert(TotalCodeMinAddr <= TotalCodeMaxAddr && "incorrect output addresses");
  size_t HotCodeSize = HotCodeMaxAddr - HotCodeMinAddr;
  size_t TotalCodeSize = TotalCodeMaxAddr - TotalCodeMinAddr;

  size_t HugePage2MB = 2 << 20;
  outs() << format("  Hot code takes %.2lf%% of binary (%zu bytes out of %zu, "
                   "%.2lf huge pages)\n",
                   100.0 * HotCodeSize / TotalCodeSize, HotCodeSize,
                   TotalCodeSize, double(HotCodeSize) / HugePage2MB);

  // Stats related to expected cache performance
  std::unordered_map<BinaryBasicBlock *, uint64_t> BBAddr;
  std::unordered_map<BinaryBasicBlock *, uint64_t> BBSize;
  extractBasicBlockInfo(BFs, BBAddr, BBSize);

  outs() << "  Expected i-TLB cache hit ratio: "
         << format("%.2lf%%\n", expectedCacheHitRatio(BFs, BBAddr, BBSize));

  outs() << "  TSP score: "
         << format("%.0lf\n", calcTSPScore(BFs, BBAddr, BBSize));

  outs() << "  ExtTSP score: "
         << format("%.0lf\n", calcExtTSPScore(BFs, BBAddr, BBSize));
}
