//===- bolt/Profile/Heatmap.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PROFILE_HEATMAP_H
#define BOLT_PROFILE_HEATMAP_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <map>
#include <vector>

namespace llvm {
class raw_ostream;

namespace bolt {

/// Struct representing a section name and its address range in the binary.
struct SectionNameAndRange {
  StringRef Name;
  uint64_t BeginAddress;
  uint64_t EndAddress;
};

class Heatmap {
  /// Number of bytes per entry in the heat map.
  size_t BucketSize;

  /// Minimum address that is considered to be valid.
  uint64_t MinAddress;

  /// Maximum address that is considered to be valid.
  uint64_t MaxAddress;

  /// Count invalid ranges.
  uint64_t NumSkippedRanges{0};

  /// Map buckets to the number of samples.
  std::map<uint64_t, uint64_t> Map;

  /// Map section names to their address range.
  const std::vector<SectionNameAndRange> TextSections;

public:
  explicit Heatmap(uint64_t BucketSize = 4096, uint64_t MinAddress = 0,
                   uint64_t MaxAddress = std::numeric_limits<uint64_t>::max(),
                   std::vector<SectionNameAndRange> TextSections = {})
      : BucketSize(BucketSize), MinAddress(MinAddress), MaxAddress(MaxAddress),
        TextSections(TextSections) {}

  inline bool ignoreAddress(uint64_t Address) const {
    return (Address > MaxAddress) || (Address < MinAddress);
  }

  /// Register a single sample at \p Address.
  void registerAddress(uint64_t Address) {
    if (!ignoreAddress(Address))
      ++Map[Address / BucketSize];
  }

  /// Register \p Count samples at [\p StartAddress, \p EndAddress ].
  void registerAddressRange(uint64_t StartAddress, uint64_t EndAddress,
                            uint64_t Count);

  /// Return the number of ranges that failed to register.
  uint64_t getNumInvalidRanges() const { return NumSkippedRanges; }

  void print(StringRef FileName) const;

  void print(raw_ostream &OS) const;

  void printCDF(StringRef FileName) const;

  void printCDF(raw_ostream &OS) const;

  void printSectionHotness(StringRef Filename) const;

  void printSectionHotness(raw_ostream &OS) const;

  size_t size() const { return Map.size(); }
};

} // namespace bolt
} // namespace llvm

#endif
