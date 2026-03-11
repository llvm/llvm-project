//===- BPSectionOrdererBase.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COMMON_BPSECTION_ORDERER_BASE_H
#define LLD_COMMON_BPSECTION_ORDERER_BASE_H

#include "llvm/Support/GlobPattern.h"
#include <optional>
#include <string>

namespace lld {

/// Specifies a glob-based compression sort group for balanced partitioning.
struct BPCompressionSortSpec {
  const llvm::GlobPattern glob;
  const std::string globString;
  const unsigned layoutPriority;
  // nullopt means positional priority (last match wins among positional specs).
  // Explicit matchPriority always beats positional.
  const std::optional<unsigned> matchPriority;

  BPCompressionSortSpec(llvm::GlobPattern glob, std::string globString,
                        unsigned layoutPriority,
                        std::optional<unsigned> matchPriority)
      : glob(std::move(glob)), globString(std::move(globString)),
        layoutPriority(layoutPriority), matchPriority(matchPriority) {}
};

} // namespace lld

#endif
