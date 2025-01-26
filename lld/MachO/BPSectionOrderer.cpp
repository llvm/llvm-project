//===- BPSectionOrderer.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BPSectionOrderer.h"
#include "InputSection.h"
#include "llvm/ADT/DenseMap.h"

#define DEBUG_TYPE "bp-section-orderer"

using namespace llvm;
using namespace lld::macho;

DenseMap<const InputSection *, int> lld::macho::runBalancedPartitioning(
    StringRef profilePath, bool forFunctionCompression, bool forDataCompression,
    bool compressionSortStartupFunctions, bool verbose) {

  SmallVector<std::unique_ptr<BPSectionBase>> sections;
  for (const auto *file : inputFiles) {
    for (auto *sec : file->sections) {
      for (auto &subsec : sec->subsections) {
        auto *isec = subsec.isec;
        if (!isec || isec->data.empty() || !isec->data.data())
          continue;
        sections.emplace_back(std::make_unique<BPSectionMacho>(isec));
      }
    }
  }

  auto reorderedSections = BPSectionBase::reorderSectionsByBalancedPartitioning(
      profilePath, forFunctionCompression, forDataCompression,
      compressionSortStartupFunctions, verbose, sections);

  DenseMap<const InputSection *, int> result;
  for (const auto &[sec, priority] : reorderedSections) {
    result.try_emplace(
        static_cast<const InputSection *>(
            static_cast<const BPSectionMacho *>(sec)->getSection()),
        priority);
  }
  return result;
}
