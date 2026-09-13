//===-- Optimizer/Support/AllocationPolicy.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
//
// The policy controlling where array allocations should live: on the stack
// (fir.alloca) or on the heap (fir.allocmem). Lowering records it on the module
// so that policy-aware passes can make consistent decisions and dumped IR
// replays with the policy it was compiled with.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_SUPPORT_ALLOCATIONPOLICY_H
#define FORTRAN_OPTIMIZER_SUPPORT_ALLOCATIONPOLICY_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

namespace mlir {
class ModuleOp;
class Operation;
} // namespace mlir

namespace fir {

/// Tunables controlling where array allocations are placed. The static
/// constants below are the only definition of the defaults. Helpers that build
/// the fir.allocation_policy attribute and pass options derive their defaults
/// from them.
struct AllocationPolicy {
  static constexpr bool stackArraysDefault = false;
  static constexpr std::uint64_t smallArrayThresholdBytesDefault = 1024;
  static constexpr std::uint64_t totalStackLimitBytesDefault =
      4ull * 1024 * 1024;

  /// Place all array allocations on the stack when possible (-fstack-arrays).
  /// When false, use the size-based policy described by the fields below.
  bool stackArrays = stackArraysDefault;
  /// Constant-size arrays up to this many bytes are considered "small".
  std::uint64_t smallArrayThresholdBytes = smallArrayThresholdBytesDefault;
  /// Per-function budget (in bytes) for small arrays placed on the stack.
  std::uint64_t totalStackLimitBytes = totalStackLimitBytesDefault;
};

/// Desired placement for an array allocation.
enum class AllocationPlacement {
  /// The allocation should live on the stack (fir.alloca).
  Stack,
  /// The allocation should live on the heap (fir.allocmem).
  Heap,
  /// The allocation should be left where it currently is.
  Leave,
};

/// Facts about an array allocation that are known before it is created. This is
/// the input to shouldAllocateOnStack, which lets code that generates
/// temporaries pick the right kind of allocation upfront instead of relying on
/// the allocation-placement pass to fix it up afterwards. Deciding upfront is
/// preferable when the generator also emits the deallocation, because the
/// lifetime is then known by construction and does not have to be proven.
struct PendingAllocationInfo {
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

/// Facts about a single existing array allocation used to decide its placement.
struct AllocationInfo : PendingAllocationInfo {
  /// The allocation operation (fir.alloca or fir.allocmem).
  mlir::Operation *op = nullptr;
  /// True if the allocation currently lives on the stack (fir.alloca).
  bool isCurrentlyOnStack = false;
};

/// Size-based placement policy, usable before the allocation is created.
/// Decides whether an allocation described by \p info should live on the stack,
/// given the \p policy in effect and the per-function stack bytes already
/// committed to the stack (\p stackBytesUsed).
bool shouldAllocateOnStack(const PendingAllocationInfo &info,
                           const AllocationPolicy &policy,
                           std::size_t stackBytesUsed);

/// Decide where an existing allocation described by \p info should live given
/// the \p policy in effect and the per-function stack bytes already committed
/// to the stack (\p stackBytesUsed). The caller is responsible for updating \p
/// stackBytesUsed based on the returned decision.
AllocationPlacement decideAllocationPlacement(const AllocationInfo &info,
                                              const AllocationPolicy &policy,
                                              std::size_t stackBytesUsed);

/// Let a pass option override one field of the policy recorded on the module.
/// Only an option that was set explicitly (in a pass pipeline string or on the
/// command line) overrides it; an option left at its default value does not, so
/// that the module attribute stays authoritative in a normal compilation and
/// tests can still pin a single field without restating the whole policy.
template <typename FieldT, typename OptionT>
void overrideIfExplicitlySet(FieldT &field, const OptionT &option) {
  if (option.hasValue())
    field = static_cast<FieldT>(option);
}

/// Placement decision hook. Has the same signature as decideAllocationPlacement
/// so the policy can be fully overridden (e.g. different thresholds inside
/// device routines or parallel regions); a hook may adjust the policy and
/// delegate to decideAllocationPlacement.
using AllocationPlacementHook = std::function<AllocationPlacement(
    const AllocationInfo & /*info*/, const AllocationPolicy & /*policy*/,
    std::size_t /*stackBytesUsed*/)>;

/// Build the policy described by the command line options above, taking the
/// -fstack-arrays part of it from \p stackArrays. Lowering records the result
/// on the module with setAllocationPolicy so that passes consulting the policy
/// use the same values.
AllocationPolicy getCommandLineAllocationPolicy(bool stackArrays);

/// Record \p policy on \p mod as a fir.allocation_policy attribute, replacing
/// any policy already recorded there.
void setAllocationPolicy(mlir::ModuleOp mod, const AllocationPolicy &policy);

/// Get the policy recorded on \p mod, or the defaults if none was recorded.
AllocationPolicy getAllocationPolicy(mlir::ModuleOp mod);

/// Get the policy in effect for \p op, which is the one recorded on its
/// enclosing ModuleOp. Returns the defaults if \p op is not inside a module or
/// if no policy was recorded.
AllocationPolicy getAllocationPolicy(mlir::Operation *op);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_ALLOCATIONPOLICY_H
