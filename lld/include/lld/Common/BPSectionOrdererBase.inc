//===- BPSectionOrdererBase.inc ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the common BPSectionOrderer interface using the Curiously
// Recurring Template Pattern and dispatches to the BalancedPartitioning
// algorithm implemented in LLVMSupport. The optimized section layout attempts
// to group similar sections together (resulting in a smaller compressed app
// size) and utilize a temporal profile file to reduce page faults during
// program startup.
//
// Clients should derive from BPOrderer, providing concrete implementations for
// section and symbol representations. Include this file in a .cpp file to
// specialize the template for the derived class.
//
//===----------------------------------------------------------------------===//

#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/CachedHashString.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/BalancedPartitioning.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <memory>
#include <optional>
#include <set>

#define DEBUG_TYPE "bp-section-orderer"

using namespace llvm;
using namespace lld;

namespace lld {
template <class D> struct BPOrdererTraits;

template <class D> struct BPOrderer {
  using Section = typename BPOrdererTraits<D>::Section;
  using Defined = typename BPOrdererTraits<D>::Defined;

  // Compute a section order using the Balanced Partitioning algorithm.
  //
  // * for*Compresion: Improve Lempel-Ziv compression by grouping
  //   similar sections together.
  // * profilePath: Utilize a temporal profile file to reduce page faults during
  //   program startup.
  // * compressionSortStartupFunctions: if profilePath is specified, allocate
  //   extra utility vertices to prioritize nearby function similarity.
  auto computeOrder(llvm::StringRef profilePath, bool forFunctionCompression,
                    bool forDataCompression,
                    bool compressionSortStartupFunctions, bool verbose,
                    llvm::ArrayRef<Section *> sections,
                    const DenseMap<CachedHashStringRef, std::set<unsigned>>
                        &rootSymbolToSectionIdxs)
      -> llvm::DenseMap<const Section *, int>;

  std::optional<StringRef> static getResolvedLinkageName(StringRef name) {
    return {};
  }
};
} // namespace lld

using UtilityNodes = SmallVector<BPFunctionNode::UtilityNodeT>;

template <class D>
static SmallVector<std::pair<unsigned, UtilityNodes>> getUnsForCompression(
    ArrayRef<const typename D::Section *> sections,
    const DenseMap<const void *, uint64_t> &sectionToIdx,
    ArrayRef<unsigned> sectionIdxs,
    DenseMap<unsigned, SmallVector<unsigned, 0>> *duplicateSectionIdxs,
    BPFunctionNode::UtilityNodeT &maxUN) {
  TimeTraceScope timeScope("Build nodes for compression");

  SmallVector<std::pair<unsigned, SmallVector<uint64_t>>> sectionHashes;
  sectionHashes.reserve(sectionIdxs.size());
  SmallVector<uint64_t> hashes;

  for (unsigned sectionIdx : sectionIdxs) {
    const auto *isec = sections[sectionIdx];
    D::getSectionHashes(*isec, hashes, sectionToIdx);
    sectionHashes.emplace_back(sectionIdx, std::move(hashes));
    hashes.clear();
  }

  MapVector<uint64_t, unsigned> hashFrequency;
  for (auto &[sectionIdx, hashes] : sectionHashes)
    for (auto hash : hashes)
      ++hashFrequency[hash];

  if (duplicateSectionIdxs) {
    // Merge sections that are nearly identical
    SmallVector<std::pair<unsigned, SmallVector<uint64_t>>> newSectionHashes;
    DenseMap<uint64_t, unsigned> wholeHashToSectionIdx;
    unsigned threshold = sectionHashes.size() > 10000 ? 5 : 0;
    for (auto &[sectionIdx, hashes] : sectionHashes) {
      uint64_t wholeHash = 0;
      for (auto hash : hashes)
        if (hashFrequency[hash] > threshold)
          wholeHash ^= hash;
      auto [it, wasInserted] =
          wholeHashToSectionIdx.insert(std::make_pair(wholeHash, sectionIdx));
      if (wasInserted) {
        newSectionHashes.emplace_back(sectionIdx, hashes);
      } else {
        (*duplicateSectionIdxs)[it->getSecond()].push_back(sectionIdx);
      }
    }
    sectionHashes = newSectionHashes;

    // Recompute hash frequencies
    hashFrequency.clear();
    for (auto &[sectionIdx, hashes] : sectionHashes)
      for (auto hash : hashes)
        ++hashFrequency[hash];
  }

  // Filter rare and common hashes and assign each a unique utility node that
  // doesn't conflict with the trace utility nodes
  DenseMap<uint64_t, BPFunctionNode::UtilityNodeT> hashToUN;
  for (auto &[hash, frequency] : hashFrequency) {
    if (frequency <= 1 || frequency * 2 > sectionHashes.size())
      continue;
    hashToUN[hash] = ++maxUN;
  }

  SmallVector<std::pair<unsigned, UtilityNodes>> sectionUns;
  for (auto &[sectionIdx, hashes] : sectionHashes) {
    UtilityNodes uns;
    for (auto &hash : hashes) {
      auto it = hashToUN.find(hash);
      if (it != hashToUN.end())
        uns.push_back(it->second);
    }
    sectionUns.emplace_back(sectionIdx, uns);
  }
  return sectionUns;
}

/// Symbols can be appended with "(.__uniq.xxxx)?.llvm.yyyy" where "xxxx" and
/// "yyyy" are numbers that could change between builds. We need to use the
/// root symbol name before this suffix so these symbols can be matched with
/// profiles which may have different suffixes.
inline StringRef getRootSymbol(StringRef name) {
  auto [P0, S0] = name.rsplit(".llvm.");
  auto [P1, S1] = P0.rsplit(".__uniq.");
  return P1;
}

template <class D>
auto BPOrderer<D>::computeOrder(
    StringRef profilePath, bool forFunctionCompression, bool forDataCompression,
    bool compressionSortStartupFunctions, bool verbose,
    ArrayRef<Section *> sections,
    const DenseMap<CachedHashStringRef, std::set<unsigned>>
        &rootSymbolToSectionIdxs) -> DenseMap<const Section *, int> {
  TimeTraceScope timeScope("Setup Balanced Partitioning");
  DenseMap<const void *, uint64_t> sectionToIdx;
  for (auto [i, isec] : llvm::enumerate(sections))
    sectionToIdx.try_emplace(isec, i);

  BPFunctionNode::UtilityNodeT maxUN = 0;
  DenseMap<unsigned, UtilityNodes> startupSectionIdxUNs;
  // Used to define the initial order for startup functions.
  DenseMap<unsigned, size_t> sectionIdxToTimestamp;
  std::unique_ptr<InstrProfReader> reader;
  if (!profilePath.empty()) {
    auto fs = vfs::getRealFileSystem();
    auto readerOrErr = InstrProfReader::create(profilePath, *fs);
    lld::checkError(readerOrErr.takeError());

    reader = std::move(readerOrErr.get());
    for (auto &entry : *reader) {
      // Read all entries
      (void)entry;
    }
    auto &traces = reader->getTemporalProfTraces();

    DenseMap<unsigned, BPFunctionNode::UtilityNodeT> sectionIdxToFirstUN;
    for (size_t traceIdx = 0; traceIdx < traces.size(); traceIdx++) {
      uint64_t currentSize = 0, cutoffSize = 1;
      size_t cutoffTimestamp = 1;
      auto &trace = traces[traceIdx].FunctionNameRefs;
      for (size_t timestamp = 0; timestamp < trace.size(); timestamp++) {
        auto [_, parsedFuncName] = getParsedIRPGOName(
            reader->getSymtab().getFuncOrVarName(trace[timestamp]));
        parsedFuncName = getRootSymbol(parsedFuncName);

        auto sectionIdxsIt =
            rootSymbolToSectionIdxs.find(CachedHashStringRef(parsedFuncName));
        if (sectionIdxsIt == rootSymbolToSectionIdxs.end())
          continue;
        auto &sectionIdxs = sectionIdxsIt->second;
        // If the same symbol is found in multiple sections, they might be
        // identical, so we arbitrarily use the size from the first section.
        currentSize += D::getSize(*sections[*sectionIdxs.begin()]);

        // Since BalancedPartitioning is sensitive to the initial order, we need
        // to explicitly define it to be ordered by earliest timestamp.
        for (unsigned sectionIdx : sectionIdxs) {
          auto [it, wasInserted] =
              sectionIdxToTimestamp.try_emplace(sectionIdx, timestamp);
          if (!wasInserted)
            it->getSecond() = std::min<size_t>(it->getSecond(), timestamp);
        }

        if (timestamp >= cutoffTimestamp || currentSize >= cutoffSize) {
          ++maxUN;
          cutoffSize = 2 * currentSize;
          cutoffTimestamp = 2 * cutoffTimestamp;
        }
        for (unsigned sectionIdx : sectionIdxs)
          sectionIdxToFirstUN.try_emplace(sectionIdx, maxUN);
      }
      for (auto &[sectionIdx, firstUN] : sectionIdxToFirstUN)
        for (auto un = firstUN; un <= maxUN; ++un)
          startupSectionIdxUNs[sectionIdx].push_back(un);
      ++maxUN;
      sectionIdxToFirstUN.clear();
    }
  }

  SmallVector<unsigned> sectionIdxsForFunctionCompression,
      sectionIdxsForDataCompression;
  for (unsigned sectionIdx = 0; sectionIdx < sections.size(); sectionIdx++) {
    if (startupSectionIdxUNs.count(sectionIdx))
      continue;
    const auto *isec = sections[sectionIdx];
    if (D::isCodeSection(*isec)) {
      if (forFunctionCompression)
        sectionIdxsForFunctionCompression.push_back(sectionIdx);
    } else {
      if (forDataCompression)
        sectionIdxsForDataCompression.push_back(sectionIdx);
    }
  }

  if (compressionSortStartupFunctions) {
    SmallVector<unsigned> startupIdxs;
    for (auto &[sectionIdx, uns] : startupSectionIdxUNs)
      startupIdxs.push_back(sectionIdx);
    auto unsForStartupFunctionCompression =
        getUnsForCompression<D>(sections, sectionToIdx, startupIdxs,
                                /*duplicateSectionIdxs=*/nullptr, maxUN);
    for (auto &[sectionIdx, compressionUns] :
         unsForStartupFunctionCompression) {
      auto &uns = startupSectionIdxUNs[sectionIdx];
      uns.append(compressionUns);
      llvm::sort(uns);
      uns.erase(std::unique(uns.begin(), uns.end()), uns.end());
    }
  }

  // Map a section index (order directly) to a list of duplicate section indices
  // (not ordered directly).
  DenseMap<unsigned, SmallVector<unsigned, 0>> duplicateSectionIdxs;
  auto unsForFunctionCompression = getUnsForCompression<D>(
      sections, sectionToIdx, sectionIdxsForFunctionCompression,
      &duplicateSectionIdxs, maxUN);
  auto unsForDataCompression = getUnsForCompression<D>(
      sections, sectionToIdx, sectionIdxsForDataCompression,
      &duplicateSectionIdxs, maxUN);

  std::vector<BPFunctionNode> nodesForStartup, nodesForFunctionCompression,
      nodesForDataCompression;
  for (auto &[sectionIdx, uns] : startupSectionIdxUNs)
    nodesForStartup.emplace_back(sectionIdx, uns);
  for (auto &[sectionIdx, uns] : unsForFunctionCompression)
    nodesForFunctionCompression.emplace_back(sectionIdx, uns);
  for (auto &[sectionIdx, uns] : unsForDataCompression)
    nodesForDataCompression.emplace_back(sectionIdx, uns);

  // Use the first timestamp to define the initial order for startup nodes.
  llvm::sort(nodesForStartup, [&sectionIdxToTimestamp](auto &L, auto &R) {
    return std::make_pair(sectionIdxToTimestamp[L.Id], L.Id) <
           std::make_pair(sectionIdxToTimestamp[R.Id], R.Id);
  });
  // Sort compression nodes by their Id (which is the section index) because the
  // input linker order tends to be not bad.
  llvm::sort(nodesForFunctionCompression,
             [](auto &L, auto &R) { return L.Id < R.Id; });
  llvm::sort(nodesForDataCompression,
             [](auto &L, auto &R) { return L.Id < R.Id; });

  {
    TimeTraceScope timeScope("Balanced Partitioning");
    BalancedPartitioningConfig config;
    BalancedPartitioning bp(config);
    bp.run(nodesForStartup);
    bp.run(nodesForFunctionCompression);
    bp.run(nodesForDataCompression);
  }

  unsigned numStartupSections = 0;
  unsigned numCodeCompressionSections = 0;
  unsigned numDuplicateCodeSections = 0;
  unsigned numDataCompressionSections = 0;
  unsigned numDuplicateDataSections = 0;
  SetVector<const Section *> orderedSections;
  // Order startup functions,
  for (auto &node : nodesForStartup) {
    const auto *isec = sections[node.Id];
    if (orderedSections.insert(isec))
      ++numStartupSections;
  }
  // then functions for compression,
  for (auto &node : nodesForFunctionCompression) {
    const auto *isec = sections[node.Id];
    if (orderedSections.insert(isec))
      ++numCodeCompressionSections;

    auto It = duplicateSectionIdxs.find(node.Id);
    if (It == duplicateSectionIdxs.end())
      continue;
    for (auto dupSecIdx : It->getSecond()) {
      const auto *dupIsec = sections[dupSecIdx];
      if (orderedSections.insert(dupIsec))
        ++numDuplicateCodeSections;
    }
  }
  // then data for compression.
  for (auto &node : nodesForDataCompression) {
    const auto *isec = sections[node.Id];
    if (orderedSections.insert(isec))
      ++numDataCompressionSections;
    auto It = duplicateSectionIdxs.find(node.Id);
    if (It == duplicateSectionIdxs.end())
      continue;
    for (auto dupSecIdx : It->getSecond()) {
      const auto *dupIsec = sections[dupSecIdx];
      if (orderedSections.insert(dupIsec))
        ++numDuplicateDataSections;
    }
  }

  if (verbose) {
    unsigned numTotalOrderedSections =
        numStartupSections + numCodeCompressionSections +
        numDuplicateCodeSections + numDataCompressionSections +
        numDuplicateDataSections;
    dbgs()
        << "Ordered " << numTotalOrderedSections
        << " sections using balanced partitioning:\n  Functions for startup: "
        << numStartupSections
        << "\n  Functions for compression: " << numCodeCompressionSections
        << "\n  Duplicate functions: " << numDuplicateCodeSections
        << "\n  Data for compression: " << numDataCompressionSections
        << "\n  Duplicate data: " << numDuplicateDataSections << "\n";

    if (!profilePath.empty()) {
      // Evaluate this function order for startup
      StringMap<std::pair<uint64_t, uint64_t>> symbolToPageNumbers;
      const uint64_t pageSize = (1 << 14);
      uint64_t currentAddress = 0;
      for (const auto *isec : orderedSections) {
        for (auto *sym : static_cast<D *>(this)->getSymbols(*isec)) {
          uint64_t startAddress = currentAddress + D::getSymValue(*sym);
          uint64_t endAddress = startAddress + D::getSymSize(*sym);
          uint64_t firstPage = startAddress / pageSize;
          // I think the kernel might pull in a few pages when one it touched,
          // so it might be more accurate to force lastPage to be aligned by
          // 4?
          uint64_t lastPage = endAddress / pageSize;
          StringRef rootSymbol = D::getSymName(*sym);
          rootSymbol = getRootSymbol(rootSymbol);
          symbolToPageNumbers.try_emplace(rootSymbol, firstPage, lastPage);
          if (auto resolvedLinkageName = D::getResolvedLinkageName(rootSymbol))
            symbolToPageNumbers.try_emplace(resolvedLinkageName.value(),
                                            firstPage, lastPage);
        }
        currentAddress += D::getSize(*isec);
      }

      // The area under the curve F where F(t) is the total number of page
      // faults at step t.
      unsigned area = 0;
      for (auto &trace : reader->getTemporalProfTraces()) {
        SmallSet<uint64_t, 0> touchedPages;
        for (unsigned step = 0; step < trace.FunctionNameRefs.size(); step++) {
          auto traceId = trace.FunctionNameRefs[step];
          auto [Filename, ParsedFuncName] =
              getParsedIRPGOName(reader->getSymtab().getFuncOrVarName(traceId));
          ParsedFuncName = getRootSymbol(ParsedFuncName);
          auto it = symbolToPageNumbers.find(ParsedFuncName);
          if (it != symbolToPageNumbers.end()) {
            auto &[firstPage, lastPage] = it->getValue();
            for (uint64_t i = firstPage; i <= lastPage; i++)
              touchedPages.insert(i);
          }
          area += touchedPages.size();
        }
      }
      dbgs() << "Total area under the page fault curve: " << (float)area
             << "\n";
    }
  }

  DenseMap<const Section *, int> sectionPriorities;
  int prio = -orderedSections.size();
  for (const auto *isec : orderedSections)
    sectionPriorities[isec] = prio++;
  return sectionPriorities;
}
