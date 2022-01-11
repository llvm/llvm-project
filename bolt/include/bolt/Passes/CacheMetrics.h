//===- bolt/Passes/CacheMetrics.h - Instruction cache metrics ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions to show metrics of cache lines.
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_CACHEMETRICS_H
#define BOLT_PASSES_CACHEMETRICS_H

#include <cstdint>
#include <vector>

namespace llvm {
namespace bolt {
class BinaryFunction;
namespace CacheMetrics {

/// Calculate various metrics related to instruction cache performance.
void printAll(const std::vector<BinaryFunction *> &BinaryFunctions);

/// Calculate Extended-TSP metric, which quantifies the expected number of
/// i-cache misses for a given pair of basic blocks. The parameters are:
/// - SrcAddr is the address of the source block;
/// - SrcSize is the size of the source block;
/// - DstAddr is the address of the destination block;
/// - Count is the number of jumps between the pair of blocks.
double extTSPScore(uint64_t SrcAddr, uint64_t SrcSize, uint64_t DstAddr,
                   uint64_t Count);

} // namespace CacheMetrics
} // namespace bolt
} // namespace llvm

#endif // BOLT_PASSES_CACHEMETRICS_H
