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
  using Defined = elf::Defined;
};
namespace {
struct BPOrdererELF : lld::BPOrderer<BPOrdererELF> {
  DenseMap<const InputSectionBase *, Defined *> secToSym;

  static uint64_t getSize(const Section &sec) { return sec.getSize(); }
  static bool isCodeSection(const Section &sec) {
    return sec.flags & ELF::SHF_EXECINSTR;
  }
  ArrayRef<Defined *> getSymbols(const Section &sec) {
    auto it = secToSym.find(&sec);
    if (it == secToSym.end())
      return {};
    return ArrayRef(it->second);
  }

  static void
  getSectionHashes(const Section &sec, SmallVectorImpl<uint64_t> &hashes,
                   const DenseMap<const void *, uint64_t> &sectionToIdx) {
    constexpr unsigned windowSize = 4;

    // Calculate content hashes: k-mers and the last k-1 bytes.
    ArrayRef<uint8_t> data = sec.content();
    if (data.size() >= windowSize)
      for (size_t i = 0; i <= data.size() - windowSize; ++i)
        hashes.push_back(support::endian::read32le(data.data() + i));
    for (uint8_t byte : data.take_back(windowSize - 1))
      hashes.push_back(byte);

    llvm::sort(hashes);
    hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
  }

  static StringRef getSymName(const Defined &sym) { return sym.getName(); }
  static uint64_t getSymValue(const Defined &sym) { return sym.value; }
  static uint64_t getSymSize(const Defined &sym) { return sym.size; }
};
} // namespace

DenseMap<const InputSectionBase *, int> elf::runBalancedPartitioning(
    Ctx &ctx, StringRef profilePath, bool forFunctionCompression,
    bool forDataCompression, bool compressionSortStartupFunctions,
    bool verbose) {
  // Collect candidate sections and associated symbols.
  SmallVector<InputSectionBase *> sections;
  DenseMap<CachedHashStringRef, std::set<unsigned>> rootSymbolToSectionIdxs;
  BPOrdererELF orderer;

  auto addSection = [&](Symbol &sym) {
    auto *d = dyn_cast<Defined>(&sym);
    if (!d)
      return;
    auto *sec = dyn_cast_or_null<InputSectionBase>(d->section);
    if (!sec || sec->size == 0 || !orderer.secToSym.try_emplace(sec, d).second)
      return;
    rootSymbolToSectionIdxs[CachedHashStringRef(getRootSymbol(sym.getName()))]
        .insert(sections.size());
    sections.emplace_back(sec);
  };

  for (Symbol *sym : ctx.symtab->getSymbols())
    addSection(*sym);
  for (ELFFileBase *file : ctx.objectFiles)
    for (Symbol *sym : file->getLocalSymbols())
      addSection(*sym);
  return orderer.computeOrder(profilePath, forFunctionCompression,
                              forDataCompression,
                              compressionSortStartupFunctions, verbose,
                              sections, rootSymbolToSectionIdxs);
}
