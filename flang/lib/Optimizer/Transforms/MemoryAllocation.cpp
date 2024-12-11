//===- MemoryAllocation.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/MemoryUtils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

namespace fir {
#define GEN_PASS_DEF_MEMORYALLOCATIONOPT
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-memory-allocation-opt"

// Number of elements in an array does not determine where it is allocated.
static constexpr std::size_t unlimitedArraySize = ~static_cast<std::size_t>(0);

/// Return `true` if this allocation is to remain on the stack (`fir.alloca`).
/// Otherwise the allocation should be moved to the heap (`fir.allocmem`).
static inline bool
keepStackAllocation(fir::AllocaOp alloca,
                    const fir::MemoryAllocationOptOptions &options) {
  // Move all arrays and character with runtime determined size to the heap.
  if (options.dynamicArrayOnHeap && alloca.isDynamic())
    return false;
  // TODO: use data layout to reason in terms of byte size to cover all "big"
  // entities, which may be scalar derived types.
  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(alloca.getInType())) {
    if (!fir::hasDynamicSize(seqTy)) {
      std::int64_t numberOfElements = 1;
      for (std::int64_t i : seqTy.getShape()) {
        numberOfElements *= i;
        // If the count is suspicious, then don't change anything here.
        if (numberOfElements <= 0)
          return true;
      }
      // If the number of elements exceeds the threshold, move the allocation to
      // the heap.
      if (static_cast<std::size_t>(numberOfElements) >
          options.maxStackArraySize) {
        return false;
      }
    }
  }
  return true;
}

static mlir::Value genAllocmem(mlir::OpBuilder &builder, fir::AllocaOp alloca,
                               bool deallocPointsDominateAlloc) {
  mlir::Type varTy = alloca.getInType();
  auto unpackName = [](std::optional<llvm::StringRef> opt) -> llvm::StringRef {
    if (opt)
      return *opt;
    return {};
  };
  llvm::StringRef uniqName = unpackName(alloca.getUniqName());
  llvm::StringRef bindcName = unpackName(alloca.getBindcName());
  auto heap = builder.create<fir::AllocMemOp>(alloca.getLoc(), varTy, uniqName,
                                              bindcName, alloca.getTypeparams(),
                                              alloca.getShape());
  LLVM_DEBUG(llvm::dbgs() << "memory allocation opt: replaced " << alloca
                          << " with " << heap << '\n');
  return heap;
}

static void genFreemem(mlir::Location loc, mlir::OpBuilder &builder,
                       mlir::Value allocmem) {
  [[maybe_unused]] auto free = builder.create<fir::FreeMemOp>(loc, allocmem);
  LLVM_DEBUG(llvm::dbgs() << "memory allocation opt: add free " << free
                          << " for " << allocmem << '\n');
}

/// This pass can reclassify memory allocations (fir.alloca, fir.allocmem) based
/// on heuristics and settings. The intention is to allow better performance and
/// workarounds for conditions such as environments with limited stack space.
///
/// Currently, implements two conversions from stack to heap allocation.
///   1. If a stack allocation is an array larger than some threshold value
///      make it a heap allocation.
///   2. If a stack allocation is an array with a runtime evaluated size make
///      it a heap allocation.
namespace {
class MemoryAllocationOpt
    : public fir::impl::MemoryAllocationOptBase<MemoryAllocationOpt> {
public:
  MemoryAllocationOpt() {
    // Set options with default values. (See Passes.td.) Note that the
    // command-line options, e.g. dynamicArrayOnHeap,  are not set yet.
    options = {dynamicArrayOnHeap, maxStackArraySize};
  }

  MemoryAllocationOpt(bool dynOnHeap, std::size_t maxStackSize) {
    // Set options with default values. (See Passes.td.)
    options = {dynOnHeap, maxStackSize};
  }

  MemoryAllocationOpt(const fir::MemoryAllocationOptOptions &options)
      : options{options} {}

  /// Override `options` if command-line options have been set.
  inline void useCommandLineOptions() {
    if (dynamicArrayOnHeap)
      options.dynamicArrayOnHeap = dynamicArrayOnHeap;
    if (maxStackArraySize != unlimitedArraySize)
      options.maxStackArraySize = maxStackArraySize;
  }

  void runOnOperation() override {
    auto *context = &getContext();
    auto func = getOperation();
    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    useCommandLineOptions();
    LLVM_DEBUG(llvm::dbgs()
               << "dynamic arrays on heap: " << options.dynamicArrayOnHeap
               << "\nmaximum number of elements of array on stack: "
               << options.maxStackArraySize << '\n');

    // If func is a declaration, skip it.
    if (func.empty())
      return;
    auto tryReplacing = [&](fir::AllocaOp alloca) {
      bool res = !keepStackAllocation(alloca, options);
      if (res) {
        LLVM_DEBUG(llvm::dbgs()
                   << "memory allocation opt: found " << alloca << '\n');
      }
      return res;
    };
    mlir::IRRewriter rewriter(context);
    fir::replaceAllocas(rewriter, func.getOperation(), tryReplacing,
                        genAllocmem, genFreemem);
  }

private:
  fir::MemoryAllocationOptOptions options;
};
} // namespace
