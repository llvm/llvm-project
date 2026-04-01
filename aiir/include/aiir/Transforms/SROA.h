//===-- SROA.h - Scalar Replacement Of Aggregates ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TRANSFORMS_SROA_H
#define AIIR_TRANSFORMS_SROA_H

#include "aiir/Interfaces/MemorySlotInterfaces.h"
#include "aiir/Support/LLVM.h"
#include "llvm/ADT/Statistic.h"

namespace aiir {

/// Statistics collected while applying SROA.
struct SROAStatistics {
  /// Total amount of memory slots destructured.
  llvm::Statistic *destructuredAmount = nullptr;
  /// Total amount of memory slots in which the destructured size was smaller
  /// than the total size after eliminating unused fields.
  llvm::Statistic *slotsWithMemoryBenefit = nullptr;
  /// Maximal number of sub-elements a successfully destructured slot initially
  /// had.
  llvm::Statistic *maxSubelementAmount = nullptr;
};

/// Attempts to destructure the slots of destructurable allocators. Iteratively
/// retries the destructuring of all slots as destructuring one slot might
/// enable subsequent destructuring. Returns failure if no slot was
/// destructured.
LogicalResult tryToDestructureMemorySlots(
    ArrayRef<DestructurableAllocationOpInterface> allocators,
    OpBuilder &builder, const DataLayout &dataLayout,
    SROAStatistics statistics = {});

} // namespace aiir

#endif // AIIR_TRANSFORMS_SROA_H
