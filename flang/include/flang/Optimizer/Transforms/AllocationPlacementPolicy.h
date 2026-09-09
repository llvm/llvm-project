//===-- Optimizer/Transforms/AllocationPlacementPolicy.h --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines the policy used by the allocation-placement pass to
// decide whether an array allocation should live on the stack (fir.alloca) or
// on the heap (fir.allocmem). The policy is expressed with tunable thresholds
// and a per-function stack budget, and can be customized through a hook (e.g.
// with different thresholds inside device routines or parallel regions).
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_ALLOCATIONPLACEMENTPOLICY_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_ALLOCATIONPLACEMENTPOLICY_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

namespace mlir {
class Operation;
} // namespace mlir

namespace fir {

/// Desired placement for an array allocation.
enum class AllocationPlacement {
  /// The allocation should live on the stack (fir.alloca).
  Stack,
  /// The allocation should live on the heap (fir.allocmem).
  Heap,
  /// The allocation should be left where it currently is.
  Leave,
};

/// Tunable thresholds controlling where array allocations are placed.
struct AllocationPlacementThresholds {
  /// Place all array allocations on the stack when possible (-fstack-arrays).
  /// When false, use the size/kind-based placement policy.
  bool stackArrays = false;
  /// Constant-size arrays up to this many bytes are considered "small".
  std::size_t smallArrayThresholdBytes = 64;
  /// Per-function budget (in bytes) for small arrays placed on the stack.
  std::size_t totalStackLimitBytes = 4ull * 1024 * 1024;
};

/// Facts about a single array allocation used to decide its placement.
struct AllocationInfo {
  /// The allocation operation (fir.alloca or fir.allocmem).
  mlir::Operation *op = nullptr;
  /// True if the allocation currently lives on the stack (fir.alloca).
  bool isCurrentlyOnStack = false;
  /// True if the allocation is a compiler temporary (as opposed to a user
  /// variable).
  bool isTemporary = false;
  /// True if the allocation has a runtime-determined size. Note this is not the
  /// same as !byteSize: a constant-size array may have no computable byteSize
  /// (e.g. when no data layout is available), in which case it is not dynamic
  /// but its size is still unknown.
  bool isDynamic = false;
  /// The constant size of the allocation in bytes, if it can be determined.
  std::optional<std::int64_t> byteSize;
};

/// Default placement policy. Decides where \p info should live given \p
/// thresholds and the per-function stack bytes already committed to the stack
/// (\p stackBytesUsed). The caller is responsible for updating \p
/// stackBytesUsed based on the returned decision.
AllocationPlacement
decideAllocationPlacement(const AllocationInfo &info,
                          const AllocationPlacementThresholds &thresholds,
                          std::size_t stackBytesUsed);

/// Placement decision hook. Has the same signature as decideAllocationPlacement
/// so the policy can be fully overridden (e.g. different thresholds inside
/// device routines or parallel regions); a hook may adjust the thresholds and
/// delegate to decideAllocationPlacement.
using AllocationPlacementHook = std::function<AllocationPlacement(
    const AllocationInfo & /*info*/,
    const AllocationPlacementThresholds & /*thresholds*/,
    std::size_t /*stackBytesUsed*/)>;

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_ALLOCATIONPLACEMENTPOLICY_H
