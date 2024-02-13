//===- bolt/Passes/ReorderSection.h - Reorder section data ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_REORDER_DATA_H
#define BOLT_PASSES_REORDER_DATA_H

#include "bolt/Passes/BinaryPasses.h"
#include <unordered_map>

namespace llvm {
namespace bolt {

class ReorderData : public BinaryFunctionPass {
public:
  using DataOrder = std::vector<std::pair<BinaryData *, uint64_t>>;

private:
  DataOrder baseOrder(BinaryContext &BC, const BinarySection &Section) const;

  std::unordered_map<BinaryData *, uint64_t> BinaryDataCounts;

  void assignMemData(BinaryContext &BC);

  /// Sort symbols by memory profiling data execution count.  The output
  /// is a vector of [address,count] pairs.
  std::pair<DataOrder, unsigned>
  sortedByCount(BinaryContext &BC, const BinarySection &Section) const;

  std::pair<DataOrder, unsigned>
  sortedByFunc(BinaryContext &BC, const BinarySection &Section,
               std::map<uint64_t, BinaryFunction> &BFs) const;

  void printOrder(BinaryContext &BC, const BinarySection &Section,
                  DataOrder::const_iterator Begin,
                  DataOrder::const_iterator End) const;

  /// Set the ordering of the section with \p SectionName.  \p NewOrder is a
  /// vector of [old address, size] pairs.  The new symbol order is implicit
  /// in the order of the vector.
  void setSectionOrder(BinaryContext &BC, BinarySection &OutputSection,
                       DataOrder::iterator Begin, DataOrder::iterator End);

  bool markUnmoveableSymbols(BinaryContext &BC, BinarySection &Section) const;

public:
  explicit ReorderData() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "reorder-data"; }

  Error runOnFunctions(BinaryContext &BC) override;
};

} // namespace bolt
} // namespace llvm

#endif
