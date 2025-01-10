//===- BPSectionOrdererBase.h ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the common interfaces which may be used by
// BPSectionOrderer.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COMMON_BP_SECTION_ORDERER_BASE_H
#define LLD_COMMON_BP_SECTION_ORDERER_BASE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/xxhash.h"
#include <memory>
#include <optional>

namespace lld {

class BPSymbol {

public:
  virtual ~BPSymbol() = default;
  virtual llvm::StringRef getName() const = 0;
  virtual std::optional<uint64_t> getValue() const = 0;
  virtual std::optional<uint64_t> getSize() const = 0;
};

class BPSectionBase {
public:
  virtual ~BPSectionBase() = default;
  virtual uint64_t getSize() const = 0;
  virtual bool isCodeSection() const = 0;
  virtual llvm::SmallVector<std::unique_ptr<BPSymbol>> getSymbols() const = 0;
  virtual const void *getSection() const = 0;
  virtual void getSectionHashes(
      llvm::SmallVectorImpl<uint64_t> &hashes,
      const llvm::DenseMap<const void *, uint64_t> &sectionToIdx) const = 0;
  virtual std::optional<llvm::StringRef>
  getResolvedLinkageName(llvm::StringRef name) const = 0;

  /// Symbols can be appended with "(.__uniq.xxxx)?.llvm.yyyy" where "xxxx" and
  /// "yyyy" are numbers that could change between builds. We need to use the
  /// root symbol name before this suffix so these symbols can be matched with
  /// profiles which may have different suffixes.
  static llvm::StringRef getRootSymbol(llvm::StringRef Name) {
    auto [P0, S0] = Name.rsplit(".llvm.");
    auto [P1, S1] = P0.rsplit(".__uniq.");
    return P1;
  }

  static uint64_t getRelocHash(llvm::StringRef kind, uint64_t sectionIdx,
                               uint64_t offset, uint64_t addend) {
    return llvm::xxHash64((kind + ": " + llvm::Twine::utohexstr(sectionIdx) +
                           " + " + llvm::Twine::utohexstr(offset) + " + " +
                           llvm::Twine::utohexstr(addend))
                              .str());
  }

  /// Reorders sections using balanced partitioning algorithm based on profile
  /// data.
  static llvm::DenseMap<const BPSectionBase *, int>
  reorderSectionsByBalancedPartitioning(
      llvm::StringRef profilePath, bool forFunctionCompression,
      bool forDataCompression, bool compressionSortStartupFunctions,
      bool verbose,
      llvm::SmallVector<std::unique_ptr<BPSectionBase>> &inputSections);
};

} // namespace lld

#endif
