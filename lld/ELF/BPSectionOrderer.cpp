//===- BPSectionOrderer.cpp------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BPSectionOrderer.h"

using namespace llvm;
using namespace lld::elf;

void BPSectionELF::getSectionHashes(
    llvm::SmallVectorImpl<uint64_t> &hashes,
    const llvm::DenseMap<const void *, uint64_t> &sectionToIdx) const {
  constexpr unsigned windowSize = 4;

  // Calculate content hashes: k-mers and the last k-1 bytes.
  ArrayRef<uint8_t> data = isec->content();
  if (data.size() >= windowSize)
    for (size_t i = 0; i <= data.size() - windowSize; ++i)
      hashes.push_back(llvm::support::endian::read32le(data.data() + i));
  for (uint8_t byte : data.take_back(windowSize - 1))
    hashes.push_back(byte);

  llvm::sort(hashes);
  hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
}

llvm::DenseMap<const lld::elf::InputSectionBase *, int>
lld::elf::runBalancedPartitioning(Ctx &ctx, llvm::StringRef profilePath,
                                  bool forFunctionCompression,
                                  bool forDataCompression,
                                  bool compressionSortStartupFunctions,
                                  bool verbose) {
  // Collect sections from symbols and wrap as BPSectionELF instances.
  // Deduplicates sections referenced by multiple symbols.
  SmallVector<std::unique_ptr<BPSectionBase>> sections;
  DenseSet<const InputSectionBase *> seenSections;

  auto addSection = [&](Symbol &sym) {
    if (sym.getSize() == 0)
      return;
    if (auto *d = dyn_cast<Defined>(&sym))
      if (auto *sec = dyn_cast_or_null<InputSectionBase>(d->section))
        if (seenSections.insert(sec).second)
          sections.emplace_back(std::make_unique<BPSectionELF>(sec));
  };

  for (Symbol *sym : ctx.symtab->getSymbols())
    addSection(*sym);

  for (ELFFileBase *file : ctx.objectFiles)
    for (Symbol *sym : file->getLocalSymbols())
      addSection(*sym);

  auto reorderedSections = BPSectionBase::reorderSectionsByBalancedPartitioning(
      profilePath, forFunctionCompression, forDataCompression,
      compressionSortStartupFunctions, verbose, sections);

  DenseMap<const InputSectionBase *, int> result;
  for (const auto [sec, priority] : reorderedSections) {
    auto *elfSection = cast<BPSectionELF>(sec);
    result.try_emplace(
        static_cast<const InputSectionBase *>(elfSection->getSection()),
        static_cast<int>(priority));
  }
  return result;
}
