//===- BPSectionOrdererBase.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/BPSectionOrdererBase.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/BalancedPartitioning.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/VirtualFileSystem.h"

#define DEBUG_TYPE "bp-section-orderer"

using namespace llvm;
using namespace lld;

using UtilityNodes = SmallVector<BPFunctionNode::UtilityNodeT>;

static SmallVector<std::pair<unsigned, UtilityNodes>> getUnsForCompression(
    ArrayRef<const BPSectionBase *> sections,
    const DenseMap<const void *, uint64_t> &sectionToIdx,
    ArrayRef<unsigned> sectionIdxs,
    DenseMap<unsigned, SmallVector<unsigned>> *duplicateSectionIdxs,
    BPFunctionNode::UtilityNodeT &maxUN) {
  TimeTraceScope timeScope("Build nodes for compression");

  SmallVector<std::pair<unsigned, SmallVector<uint64_t>>> sectionHashes;
  sectionHashes.reserve(sectionIdxs.size());
  SmallVector<uint64_t> hashes;

  for (unsigned sectionIdx : sectionIdxs) {
    const auto *isec = sections[sectionIdx];
    isec->getSectionHashes(hashes, sectionToIdx);
    sectionHashes.emplace_back(sectionIdx, std::move(hashes));
    hashes.clear();
  }

  DenseMap<uint64_t, unsigned> hashFrequency;
  for (auto &[sectionIdx, hashes] : sectionHashes)
    for (auto hash : hashes)
      ++hashFrequency[hash];

  if (duplicateSectionIdxs) {
    // Merge sections that are nearly identical
    SmallVector<std::pair<unsigned, SmallVector<uint64_t>>> newSectionHashes;
    DenseMap<uint64_t, unsigned> wholeHashToSectionIdx;
    for (auto &[sectionIdx, hashes] : sectionHashes) {
      uint64_t wholeHash = 0;
      for (auto hash : hashes)
        if (hashFrequency[hash] > 5)
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

llvm::DenseMap<const BPSectionBase *, size_t>
BPSectionBase::reorderSectionsByBalancedPartitioning(
    size_t &highestAvailablePriority, llvm::StringRef profilePath,
    bool forFunctionCompression, bool forDataCompression,
    bool compressionSortStartupFunctions, bool verbose,
    SmallVector<std::unique_ptr<BPSectionBase>> &inputSections) {
  TimeTraceScope timeScope("Setup Balanced Partitioning");
  SmallVector<const BPSectionBase *> sections;
  DenseMap<const void *, uint64_t> sectionToIdx;
  StringMap<DenseSet<unsigned>> symbolToSectionIdxs;

  // Process input sections
  for (const auto &isec : inputSections) {
    unsigned sectionIdx = sections.size();
    sectionToIdx.try_emplace(isec->getSection(), sectionIdx);
    sections.emplace_back(isec.get());
    for (auto &sym : isec->getSymbols())
      symbolToSectionIdxs[sym->getName()].insert(sectionIdx);
  }
  StringMap<DenseSet<unsigned>> rootSymbolToSectionIdxs;
  for (auto &entry : symbolToSectionIdxs) {
    StringRef name = entry.getKey();
    auto &sectionIdxs = entry.getValue();
    name = BPSectionBase::getRootSymbol(name);
    rootSymbolToSectionIdxs[name].insert(sectionIdxs.begin(),
                                         sectionIdxs.end());
    if (auto resolvedLinkageName =
            sections[*sectionIdxs.begin()]->getResolvedLinkageName(name))
      rootSymbolToSectionIdxs[resolvedLinkageName.value()].insert(
          sectionIdxs.begin(), sectionIdxs.end());
  }

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
        auto [Filename, ParsedFuncName] = getParsedIRPGOName(
            reader->getSymtab().getFuncOrVarName(trace[timestamp]));
        ParsedFuncName = BPSectionBase::getRootSymbol(ParsedFuncName);

        auto sectionIdxsIt = rootSymbolToSectionIdxs.find(ParsedFuncName);
        if (sectionIdxsIt == rootSymbolToSectionIdxs.end())
          continue;
        auto &sectionIdxs = sectionIdxsIt->getValue();
        // If the same symbol is found in multiple sections, they might be
        // identical, so we arbitrarily use the size from the first section.
        currentSize += sections[*sectionIdxs.begin()]->getSize();

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
    if (isec->isCodeSection()) {
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
        getUnsForCompression(sections, sectionToIdx, startupIdxs,
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
  DenseMap<unsigned, SmallVector<unsigned>> duplicateSectionIdxs;
  auto unsForFunctionCompression = getUnsForCompression(
      sections, sectionToIdx, sectionIdxsForFunctionCompression,
      &duplicateSectionIdxs, maxUN);
  auto unsForDataCompression = getUnsForCompression(
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
  SetVector<const BPSectionBase *> orderedSections;
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
        for (auto &sym : isec->getSymbols()) {
          uint64_t startAddress = currentAddress + sym->getValue().value_or(0);
          uint64_t endAddress = startAddress + sym->getSize().value_or(0);
          uint64_t firstPage = startAddress / pageSize;
          // I think the kernel might pull in a few pages when one it touched,
          // so it might be more accurate to force lastPage to be aligned by
          // 4?
          uint64_t lastPage = endAddress / pageSize;
          StringRef rootSymbol = sym->getName();
          rootSymbol = BPSectionBase::getRootSymbol(rootSymbol);
          symbolToPageNumbers.try_emplace(rootSymbol, firstPage, lastPage);
          if (auto resolvedLinkageName =
                  isec->getResolvedLinkageName(rootSymbol))
            symbolToPageNumbers.try_emplace(resolvedLinkageName.value(),
                                            firstPage, lastPage);
        }
        currentAddress += isec->getSize();
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
          ParsedFuncName = BPSectionBase::getRootSymbol(ParsedFuncName);
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

  DenseMap<const BPSectionBase *, size_t> sectionPriorities;
  for (const auto *isec : orderedSections)
    sectionPriorities[isec] = --highestAvailablePriority;
  return sectionPriorities;
}
