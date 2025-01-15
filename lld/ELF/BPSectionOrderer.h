//===- BPSectionOrderer.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file uses Balanced Partitioning to order sections to improve startup
/// time and compressed size.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_BPSECTION_ORDERER_H
#define LLD_ELF_BPSECTION_ORDERER_H

#include "InputFiles.h"
#include "InputSection.h"
#include "SymbolTable.h"
#include "lld/Common/BPSectionOrdererBase.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"

namespace lld::elf {

class InputSection;

class BPSymbolELF : public BPSymbol {
  const Symbol *sym;

public:
  explicit BPSymbolELF(const Symbol *s) : sym(s) {}

  llvm::StringRef getName() const override { return sym->getName(); }

  const Defined *asDefined() const { return llvm::dyn_cast<Defined>(sym); }

  std::optional<uint64_t> getValue() const override {
    if (auto *d = asDefined())
      return d->value;
    return {};
  }

  std::optional<uint64_t> getSize() const override {
    if (auto *d = asDefined())
      return d->size;
    return {};
  }

  InputSectionBase *getInputSection() const {
    if (auto *d = asDefined())
      return llvm::dyn_cast_or_null<InputSectionBase>(d->section);
    return nullptr;
  }

  const Symbol *getSymbol() const { return sym; }
};

class BPSectionELF : public BPSectionBase {
  const InputSectionBase *isec;

public:
  explicit BPSectionELF(const InputSectionBase *sec) : isec(sec) {}

  const void *getSection() const override { return isec; }

  uint64_t getSize() const override { return isec->getSize(); }

  bool isCodeSection() const override {
    return isec->flags & llvm::ELF::SHF_EXECINSTR;
  }

  SmallVector<std::unique_ptr<BPSymbol>> getSymbols() const override {
    SmallVector<std::unique_ptr<BPSymbol>> symbols;
    for (Symbol *sym : isec->file->getSymbols())
      if (auto *d = dyn_cast<Defined>(sym))
        if (d->size > 0 && d->section == isec)
          symbols.emplace_back(std::make_unique<BPSymbolELF>(sym));

    return symbols;
  }

  std::optional<StringRef>
  getResolvedLinkageName(llvm::StringRef name) const override {
    return {};
  }

  void getSectionHashes(llvm::SmallVectorImpl<uint64_t> &hashes,
                        const llvm::DenseMap<const void *, uint64_t>
                            &sectionToIdx) const override;

  static bool classof(const BPSectionBase *s) { return true; }
};

/// Run Balanced Partitioning to find the optimal function and data order to
/// improve startup time and compressed size.
///
/// It is important that -ffunction-sections and -fdata-sections are used to
/// ensure functions and data are in their own sections and thus can be
/// reordered.
llvm::DenseMap<const InputSectionBase *, int>
runBalancedPartitioning(Ctx &ctx, llvm::StringRef profilePath,
                        bool forFunctionCompression, bool forDataCompression,
                        bool compressionSortStartupFunctions, bool verbose);
} // namespace lld::elf

#endif
