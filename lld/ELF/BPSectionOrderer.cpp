//===- BPSectionOrderer.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BPSectionOrderer.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "lld/Common/BPSectionOrdererBase.inc"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace lld::elf;

namespace {
struct BPOrdererELF;
}
template <> struct lld::BPOrdererTraits<struct BPOrdererELF> {
  using Section = elf::InputSectionBase;
  using Symbol = elf::Symbol;
};
namespace {
struct BPOrdererELF : lld::BPOrderer<BPOrdererELF> {
  static uint64_t getSize(const Section &sec) { return sec.getSize(); }
  static bool isCodeSection(const Section &sec) {
    return sec.flags & llvm::ELF::SHF_EXECINSTR;
  }
  static SmallVector<Symbol *, 0> getSymbols(const Section &sec) {
    SmallVector<Symbol *, 0> symbols;
    for (auto *sym : sec.file->getSymbols())
      if (auto *d = llvm::dyn_cast_or_null<Defined>(sym))
        if (d->size > 0 && d->section == &sec)
          symbols.emplace_back(d);
    return symbols;
  }

  std::optional<StringRef> static getResolvedLinkageName(llvm::StringRef name) {
    return {};
  }

  static void
  getSectionHashes(const Section &sec, llvm::SmallVectorImpl<uint64_t> &hashes,
                   const llvm::DenseMap<const void *, uint64_t> &sectionToIdx) {
    constexpr unsigned windowSize = 4;

    // Calculate content hashes: k-mers and the last k-1 bytes.
    ArrayRef<uint8_t> data = sec.content();
    if (data.size() >= windowSize)
      for (size_t i = 0; i <= data.size() - windowSize; ++i)
        hashes.push_back(llvm::support::endian::read32le(data.data() + i));
    for (uint8_t byte : data.take_back(windowSize - 1))
      hashes.push_back(byte);

    llvm::sort(hashes);
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
  }

  static llvm::StringRef getSymName(const Symbol &sym) { return sym.getName(); }
  static uint64_t getSymValue(const Symbol &sym) {
    if (auto *d = dyn_cast<Defined>(&sym))
      return d->value;
    return 0;
  }
  static uint64_t getSymSize(const Symbol &sym) {
    if (auto *d = dyn_cast<Defined>(&sym))
      return d->size;
    return 0;
  }
};
} // namespace

DenseMap<const InputSectionBase *, int> elf::runBalancedPartitioning(
    Ctx &ctx, StringRef profilePath, bool forFunctionCompression,
    bool forDataCompression, bool compressionSortStartupFunctions,
    bool verbose) {
  // Collect candidate sections and associated symbols.
  SmallVector<InputSectionBase *> sections;
  DenseMap<CachedHashStringRef, DenseSet<unsigned>> rootSymbolToSectionIdxs;
  DenseSet<const InputSectionBase *> seenSections;

  auto addSection = [&](Symbol &sym) {
    auto *d = dyn_cast<Defined>(&sym);
    if (!d || d->size == 0)
      return;
    auto *sec = dyn_cast_or_null<InputSectionBase>(d->section);
    if (sec && seenSections.insert(sec).second) {
      rootSymbolToSectionIdxs[CachedHashStringRef(getRootSymbol(sym.getName()))]
          .insert(sections.size());
      sections.emplace_back(sec);
    }
  };

  for (Symbol *sym : ctx.symtab->getSymbols())
    addSection(*sym);
  for (ELFFileBase *file : ctx.objectFiles)
    for (Symbol *sym : file->getLocalSymbols())
      addSection(*sym);

  return BPOrdererELF::computeOrder(profilePath, forFunctionCompression,
                                    forDataCompression,
                                    compressionSortStartupFunctions, verbose,
                                    sections, rootSymbolToSectionIdxs);
}
