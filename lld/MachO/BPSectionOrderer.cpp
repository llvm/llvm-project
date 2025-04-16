//===- BPSectionOrderer.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BPSectionOrderer.h"
#include "InputSection.h"
#include "Relocations.h"
#include "Symbols.h"
#include "lld/Common/BPSectionOrdererBase.inc"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/xxhash.h"

#define DEBUG_TYPE "bp-section-orderer"

using namespace llvm;
using namespace lld::macho;

namespace {
struct BPOrdererMachO;
}
template <> struct lld::BPOrdererTraits<struct BPOrdererMachO> {
  using Section = macho::InputSection;
  using Defined = macho::Defined;
};
namespace {
struct BPOrdererMachO : lld::BPOrderer<BPOrdererMachO> {
  static uint64_t getSize(const Section &sec) { return sec.getSize(); }
  static bool isCodeSection(const Section &sec) {
    return macho::isCodeSection(&sec);
  }
  static ArrayRef<Defined *> getSymbols(const Section &sec) {
    return sec.symbols;
  }

  // Linkage names can be prefixed with "_" or "l_" on Mach-O. See
  // Mangler::getNameWithPrefix() for details.
  std::optional<StringRef> static getResolvedLinkageName(llvm::StringRef name) {
    if (name.consume_front("_") || name.consume_front("l_"))
      return name;
    return {};
  }

  static void
  getSectionHashes(const Section &sec, llvm::SmallVectorImpl<uint64_t> &hashes,
                   const llvm::DenseMap<const void *, uint64_t> &sectionToIdx) {
    constexpr unsigned windowSize = 4;

    // Calculate content hashes: k-mers and the last k-1 bytes.
    ArrayRef<uint8_t> data = sec.data;
    if (data.size() >= windowSize)
      for (size_t i = 0; i <= data.size() - windowSize; ++i)
        hashes.push_back(llvm::support::endian::read32le(data.data() + i));
    for (uint8_t byte : data.take_back(windowSize - 1))
      hashes.push_back(byte);

    // Calculate relocation hashes
    for (const auto &r : sec.relocs) {
      if (r.length == 0 || r.referent.isNull() || r.offset >= data.size())
        continue;

      uint64_t relocHash = getRelocHash(r, sectionToIdx);
      uint32_t start = (r.offset < windowSize) ? 0 : r.offset - windowSize + 1;
      for (uint32_t i = start; i < r.offset + r.length; i++) {
        auto window = data.drop_front(i).take_front(windowSize);
        hashes.push_back(xxh3_64bits(window) ^ relocHash);
      }
    }

    llvm::sort(hashes);
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
  }

  static llvm::StringRef getSymName(const Defined &sym) {
    return sym.getName();
  }
  static uint64_t getSymValue(const Defined &sym) { return sym.value; }
  static uint64_t getSymSize(const Defined &sym) { return sym.size; }

private:
  static uint64_t
  getRelocHash(const Reloc &reloc,
               const llvm::DenseMap<const void *, uint64_t> &sectionToIdx) {
    auto *isec = reloc.getReferentInputSection();
    std::optional<uint64_t> sectionIdx;
    if (auto it = sectionToIdx.find(isec); it != sectionToIdx.end())
      sectionIdx = it->second;
    uint64_t kind = -1, value = 0;
    if (isec)
      kind = uint64_t(isec->kind());

    if (auto *sym = reloc.referent.dyn_cast<Symbol *>()) {
      kind = (kind << 8) | uint8_t(sym->kind());
      if (auto *d = llvm::dyn_cast<Defined>(sym))
        value = d->value;
    }
    return llvm::stable_hash_combine(kind, sectionIdx.value_or(0), value,
                                     reloc.addend);
  }
};
} // namespace

DenseMap<const InputSection *, int> lld::macho::runBalancedPartitioning(
    StringRef profilePath, bool forFunctionCompression, bool forDataCompression,
    bool compressionSortStartupFunctions, bool verbose) {
  // Collect candidate sections and associated symbols.
  SmallVector<InputSection *> sections;
  DenseMap<CachedHashStringRef, std::set<unsigned>> rootSymbolToSectionIdxs;
  for (const auto *file : inputFiles) {
    for (auto *sec : file->sections) {
      for (auto &subsec : sec->subsections) {
        auto *isec = subsec.isec;
        if (!isec || isec->data.empty())
          continue;
        size_t idx = sections.size();
        sections.emplace_back(isec);
        for (auto *sym : BPOrdererMachO::getSymbols(*isec)) {
          auto rootName = getRootSymbol(sym->getName());
          rootSymbolToSectionIdxs[CachedHashStringRef(rootName)].insert(idx);
          if (auto linkageName =
                  BPOrdererMachO::getResolvedLinkageName(rootName))
            rootSymbolToSectionIdxs[CachedHashStringRef(*linkageName)].insert(
                idx);
        }
      }
    }
  }

  return BPOrdererMachO().computeOrder(profilePath, forFunctionCompression,
                                       forDataCompression,
                                       compressionSortStartupFunctions, verbose,
                                       sections, rootSymbolToSectionIdxs);
}
