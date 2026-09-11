//===-- AllocationPolicy.cpp ----------------------------------------------===//
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

#include "flang/Optimizer/Support/AllocationPolicy.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/CommandLine.h"

static constexpr const char *allocationPolicyName = "fir.allocation_policy";

static llvm::cl::opt<std::uint64_t> allocationPlacementSmallArraySize(
    "allocation-placement-small-array-size",
    llvm::cl::desc(
        "constant-size arrays up to <size> bytes are placed on the stack "
        "by the allocation-placement pass and by the copy-in inlining"),
    llvm::cl::init(fir::AllocationPolicy::smallArrayThresholdBytesDefault),
    llvm::cl::Hidden);

static llvm::cl::opt<std::uint64_t> allocationPlacementStackLimit(
    "allocation-placement-stack-limit",
    llvm::cl::desc(
        "per-function budget in bytes for small arrays placed on the stack "
        "by the allocation-placement pass"),
    llvm::cl::init(fir::AllocationPolicy::totalStackLimitBytesDefault),
    llvm::cl::Hidden);

bool fir::shouldAllocateOnStack(const PendingAllocationInfo &info,
                                const AllocationPolicy &policy,
                                std::size_t stackBytesUsed) {
  // -fstack-arrays: put everything on the stack (best effort). For existing
  // allocations, the heap-to-stack conversion still only happens where it is
  // provably safe.
  if (policy.stackArrays)
    return true;

  // Runtime-sized arrays (automatic arrays, dynamic temporaries) go on the
  // heap.
  if (info.isDynamic)
    return false;

  // Without a known constant size we cannot reason about thresholds: stay on
  // the heap, which is always valid.
  if (!info.byteSize)
    return false;

  // Constant-size user variables always go on the stack.
  if (!info.isTemporary)
    return true;

  auto size = static_cast<std::size_t>(*info.byteSize);
  // Small array temporaries go on the stack while the per-function budget
  // allows it; bigger ones go on the heap.
  return size <= policy.smallArrayThresholdBytes &&
         stackBytesUsed + size <= policy.totalStackLimitBytes;
}

fir::AllocationPlacement
fir::decideAllocationPlacement(const AllocationInfo &info,
                               const AllocationPolicy &policy,
                               std::size_t stackBytesUsed) {
  using P = fir::AllocationPlacement;

  // An allocation that is not known to be dynamic but whose size cannot be
  // determined cannot be reasoned about: leave it where it is instead of
  // moving it based on a size that is not available.
  if (!policy.stackArrays && !info.isDynamic && !info.byteSize)
    return P::Leave;

  // Translate the "should this be on the stack" decision into a placement,
  // accounting for where the allocation currently lives.
  bool wantStack = fir::shouldAllocateOnStack(info, policy, stackBytesUsed);
  if (wantStack)
    return info.isCurrentlyOnStack ? P::Leave : P::Stack;
  return info.isCurrentlyOnStack ? P::Heap : P::Leave;
}

fir::AllocationPolicy fir::getCommandLineAllocationPolicy(bool stackArrays) {
  fir::AllocationPolicy policy;
  policy.stackArrays = stackArrays;
  policy.smallArrayThresholdBytes = allocationPlacementSmallArraySize;
  policy.totalStackLimitBytes = allocationPlacementStackLimit;
  return policy;
}

void fir::setAllocationPolicy(mlir::ModuleOp mod,
                              const fir::AllocationPolicy &policy) {
  mod->setAttr(allocationPolicyName, fir::AllocationPolicyAttr::get(
                                         mod.getContext(), policy.stackArrays,
                                         policy.smallArrayThresholdBytes,
                                         policy.totalStackLimitBytes));
}

fir::AllocationPolicy fir::getAllocationPolicy(mlir::ModuleOp mod) {
  auto attr =
      mod->getAttrOfType<fir::AllocationPolicyAttr>(allocationPolicyName);
  if (!attr)
    return fir::AllocationPolicy{};
  return fir::AllocationPolicy{attr.getStackArrays(),
                               attr.getSmallArrayThreshold(),
                               attr.getTotalStackLimit()};
}

fir::AllocationPolicy fir::getAllocationPolicy(mlir::Operation *op) {
  auto mod = mlir::dyn_cast<mlir::ModuleOp>(op);
  if (!mod)
    mod = op->getParentOfType<mlir::ModuleOp>();
  if (!mod)
    return fir::AllocationPolicy{};
  return getAllocationPolicy(mod);
}
