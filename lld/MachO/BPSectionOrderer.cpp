//===- BPSectionOrderer.cpp--------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BPSectionOrderer.h"
#include "InputSection.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/BalancedPartitioning.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/xxhash.h"

#define DEBUG_TYPE "bp-section-orderer"
using namespace llvm;
using namespace lld::macho;

/// Symbols can be appended with "(.__uniq.xxxx)?.llvm.yyyy" where "xxxx" and
/// "yyyy" are numbers that could change between builds. We need to use the root
/// symbol name before this suffix so these symbols can be matched with profiles
/// which may have different suffixes.
static StringRef getRootSymbol(StringRef Name) {
  auto [P0, S0] = Name.rsplit(".llvm.");
  auto [P1, S1] = P0.rsplit(".__uniq.");
  return P1;
}

static uint64_t getRelocHash(StringRef kind, uint64_t sectionIdx,
                             uint64_t offset, uint64_t addend) {
  return xxHash64((kind + ": " + Twine::utohexstr(sectionIdx) + " + " +
                   Twine::utohexstr(offset) + " + " + Twine::utohexstr(addend))
                      .str());
}

static uint64_t
getRelocHash(const Reloc &reloc,
             const DenseMap<const InputSection *, uint64_t> &sectionToIdx) {
  auto *isec = reloc.getReferentInputSection();
  std::optional<uint64_t> sectionIdx;
  auto sectionIdxIt = sectionToIdx.find(isec);
  if (sectionIdxIt != sectionToIdx.end())
    sectionIdx = sectionIdxIt->getSecond();
  std::string kind;
  if (isec)
    kind = ("Section " + Twine((uint8_t)isec->kind())).str();
  if (auto *sym = reloc.referent.dyn_cast<Symbol *>()) {
    kind += (" Symbol " + Twine((uint8_t)sym->kind())).str();
    if (auto *d = dyn_cast<Defined>(sym)) {
      if (isa_and_nonnull<CStringInputSection>(isec))
        return getRelocHash(kind, 0, isec->getOffset(d->value), reloc.addend);
      return getRelocHash(kind, sectionIdx.value_or(0), d->value, reloc.addend);
    }
  }
  return getRelocHash(kind, sectionIdx.value_or(0), 0, reloc.addend);
}

static void constructNodesForCompression(
    const SmallVector<const InputSection *> &sections,
    const DenseMap<const InputSection *, uint64_t> &sectionToIdx,
    const SmallVector<unsigned> &sectionIdxs,
    std::vector<BPFunctionNode> &nodes,
    DenseMap<unsigned, SmallVector<unsigned>> &duplicateSectionIdxs,
    BPFunctionNode::UtilityNodeT &maxUN) {
  TimeTraceScope timeScope("Build nodes for compression");

  SmallVector<std::pair<unsigned, SmallVector<uint64_t>>> sectionHashes;
  sectionHashes.reserve(sectionIdxs.size());
  SmallVector<uint64_t> hashes;
  for (unsigned sectionIdx : sectionIdxs) {
    const auto *isec = sections[sectionIdx];
    constexpr unsigned windowSize = 4;

    for (size_t i = 0; i < isec->data.size(); i++) {
      auto window = isec->data.drop_front(i).take_front(windowSize);
      hashes.push_back(xxHash64(window));
    }
    for (const auto &r : isec->relocs) {
      if (r.length == 0 || r.referent.isNull() || r.offset >= isec->data.size())
        continue;
      uint64_t relocHash = getRelocHash(r, sectionToIdx);
      uint32_t start = (r.offset < windowSize) ? 0 : r.offset - windowSize + 1;
      for (uint32_t i = start; i < r.offset + r.length; i++) {
        auto window = isec->data.drop_front(i).take_front(windowSize);
        hashes.push_back(xxHash64(window) + relocHash);
      }
    }

    llvm::sort(hashes);
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());

    sectionHashes.emplace_back(sectionIdx, hashes);
    hashes.clear();
  }

  DenseMap<uint64_t, unsigned> hashFrequency;
  for (auto &[sectionIdx, hashes] : sectionHashes)
    for (auto hash : hashes)
      ++hashFrequency[hash];

  // Merge section that are nearly identical
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
      duplicateSectionIdxs[it->getSecond()].push_back(sectionIdx);
    }
  }
  sectionHashes = newSectionHashes;

  // Recompute hash frequencies
  hashFrequency.clear();
  for (auto &[sectionIdx, hashes] : sectionHashes)
    for (auto hash : hashes)
      ++hashFrequency[hash];

  // Filter rare and common hashes and assign each a unique utility node that
  // doesn't conflict with the trace utility nodes
  DenseMap<uint64_t, BPFunctionNode::UtilityNodeT> hashToUN;
  for (auto &[hash, frequency] : hashFrequency) {
    if (frequency <= 1 || frequency * 2 > wholeHashToSectionIdx.size())
      continue;
    hashToUN[hash] = ++maxUN;
  }

  std::vector<BPFunctionNode::UtilityNodeT> uns;
  for (auto &[sectionIdx, hashes] : sectionHashes) {
    for (auto &hash : hashes) {
      auto it = hashToUN.find(hash);
      if (it != hashToUN.end())
        uns.push_back(it->second);
    }
    nodes.emplace_back(sectionIdx, uns);
    uns.clear();
  }
}

DenseMap<const InputSection *, size_t> lld::macho::runBalancedPartitioning(
    size_t &highestAvailablePriority, StringRef profilePath,
    bool forFunctionCompression, bool forDataCompression, bool verbose) {

  SmallVector<const InputSection *> sections;
  DenseMap<const InputSection *, uint64_t> sectionToIdx;
  StringMap<DenseSet<unsigned>> symbolToSectionIdxs;
  for (const auto *file : inputFiles) {
    for (auto *sec : file->sections) {
      for (auto &subsec : sec->subsections) {
        auto *isec = subsec.isec;
        if (!isec || isec->data.empty() || !isec->data.data())
          continue;
        unsigned sectionIdx = sections.size();
        sectionToIdx.try_emplace(isec, sectionIdx);
        sections.push_back(isec);
        for (Symbol *sym : isec->symbols)
          if (auto *d = dyn_cast_or_null<Defined>(sym))
            symbolToSectionIdxs[d->getName()].insert(sectionIdx);
      }
    }
  }

  StringMap<DenseSet<unsigned>> rootSymbolToSectionIdxs;
  for (auto &entry : symbolToSectionIdxs) {
    StringRef name = entry.getKey();
    auto &sectionIdxs = entry.getValue();
    name = getRootSymbol(name);
    rootSymbolToSectionIdxs[name].insert(sectionIdxs.begin(),
                                         sectionIdxs.end());
    // Linkage names can be prefixed with "_" or "l_" on Mach-O. See
    // Mangler::getNameWithPrefix() for details.
    if (name.consume_front("_") || name.consume_front("l_"))
      rootSymbolToSectionIdxs[name].insert(sectionIdxs.begin(),
                                           sectionIdxs.end());
  }

  std::vector<BPFunctionNode> nodesForStartup;
  BPFunctionNode::UtilityNodeT maxUN = 0;
  DenseMap<unsigned, SmallVector<BPFunctionNode::UtilityNodeT>>
      startupSectionIdxUNs;
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

    // Used to define the initial order for startup functions.
    DenseMap<unsigned, size_t> sectionIdxToTimestamp;
    DenseMap<unsigned, BPFunctionNode::UtilityNodeT> sectionIdxToFirstUN;
    for (size_t traceIdx = 0; traceIdx < traces.size(); traceIdx++) {
      uint64_t currentSize = 0, cutoffSize = 1;
      size_t cutoffTimestamp = 1;
      auto &trace = traces[traceIdx].FunctionNameRefs;
      for (size_t timestamp = 0; timestamp < trace.size(); timestamp++) {
        auto [Filename, ParsedFuncName] = getParsedIRPGOName(
            reader->getSymtab().getFuncOrVarName(trace[timestamp]));
        ParsedFuncName = getRootSymbol(ParsedFuncName);

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

    // These uns should already be sorted without duplicates.
    for (auto &[sectionIdx, uns] : startupSectionIdxUNs)
      nodesForStartup.emplace_back(sectionIdx, uns);

    llvm::sort(nodesForStartup, [&sectionIdxToTimestamp](auto &L, auto &R) {
      return std::make_pair(sectionIdxToTimestamp[L.Id], L.Id) <
             std::make_pair(sectionIdxToTimestamp[R.Id], R.Id);
    });
  }

  SmallVector<unsigned> sectionIdxsForFunctionCompression,
      sectionIdxsForDataCompression;
  for (unsigned sectionIdx = 0; sectionIdx < sections.size(); sectionIdx++) {
    if (startupSectionIdxUNs.count(sectionIdx))
      continue;
    const auto *isec = sections[sectionIdx];
    if (isCodeSection(isec)) {
      if (forFunctionCompression)
        sectionIdxsForFunctionCompression.push_back(sectionIdx);
    } else {
      if (forDataCompression)
        sectionIdxsForDataCompression.push_back(sectionIdx);
    }
  }

  std::vector<BPFunctionNode> nodesForFunctionCompression,
      nodesForDataCompression;
  // Map a section index (to be ordered for compression) to a list of duplicate
  // section indices (not ordered for compression).
  DenseMap<unsigned, SmallVector<unsigned>> duplicateFunctionSectionIdxs,
      duplicateDataSectionIdxs;
  constructNodesForCompression(
      sections, sectionToIdx, sectionIdxsForFunctionCompression,
      nodesForFunctionCompression, duplicateFunctionSectionIdxs, maxUN);
  constructNodesForCompression(
      sections, sectionToIdx, sectionIdxsForDataCompression,
      nodesForDataCompression, duplicateDataSectionIdxs, maxUN);

  // Sort nodes by their Id (which is the section index) because the input
  // linker order tends to be not bad
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
  SetVector<const InputSection *> orderedSections;
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

    auto It = duplicateFunctionSectionIdxs.find(node.Id);
    if (It == duplicateFunctionSectionIdxs.end())
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
    auto It = duplicateDataSectionIdxs.find(node.Id);
    if (It == duplicateDataSectionIdxs.end())
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
        for (Symbol *sym : isec->symbols) {
          if (auto *d = dyn_cast_or_null<Defined>(sym)) {
            uint64_t startAddress = currentAddress + d->value;
            uint64_t endAddress = startAddress + d->size;
            uint64_t firstPage = startAddress / pageSize;
            // I think the kernel might pull in a few pages when one it touched,
            // so it might be more accurate to force lastPage to be aligned by
            // 4?
            uint64_t lastPage = endAddress / pageSize;
            StringRef rootSymbol = d->getName();
            rootSymbol = getRootSymbol(rootSymbol);
            symbolToPageNumbers.try_emplace(rootSymbol, firstPage, lastPage);
            if (rootSymbol.consume_front("_") || rootSymbol.consume_front("l_"))
              symbolToPageNumbers.try_emplace(rootSymbol, firstPage, lastPage);
          }
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

  DenseMap<const InputSection *, size_t> sectionPriorities;
  for (const auto *isec : orderedSections)
    sectionPriorities[isec] = --highestAvailablePriority;
  return sectionPriorities;
}
