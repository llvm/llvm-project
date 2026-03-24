//===- BPSectionOrdererBase.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COMMON_BPSECTION_ORDERER_BASE_H
#define LLD_COMMON_BPSECTION_ORDERER_BASE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/GlobPattern.h"
#include <optional>

namespace lld {

/// Specifies a glob-based compression sort group for balanced partitioning.
struct BPCompressionSortSpec {
  static llvm::Expected<BPCompressionSortSpec>
  create(llvm::StringRef globString, unsigned layoutPriority,
         std::optional<unsigned> matchPriority) {
    auto glob = llvm::GlobPattern::create(globString);
    if (!glob)
      return glob.takeError();
    return BPCompressionSortSpec(std::move(*glob), globString, layoutPriority,
                                 matchPriority);
  }

  const llvm::GlobPattern glob;
  const llvm::StringRef globString;
  const unsigned layoutPriority;
  // nullopt means positional priority (last match wins among positional specs).
  // Explicit matchPriority always beats positional.
  const std::optional<unsigned> matchPriority;

private:
  BPCompressionSortSpec(llvm::GlobPattern glob, llvm::StringRef globString,
                        unsigned layoutPriority,
                        std::optional<unsigned> matchPriority)
      : glob(std::move(glob)), globString(globString),
        layoutPriority(layoutPriority), matchPriority(matchPriority) {}
};

} // namespace lld

#endif
