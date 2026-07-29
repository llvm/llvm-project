//===- AllocationPlacement.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass decides, for each array allocation in a function, whether it should
// live on the stack (fir.alloca) or on the heap (fir.allocmem), and rewrites it
// accordingly. The decision is delegated to the policy in
// AllocationPlacementPolicy.h. Two rewrite engines are reused:
//   - stack-to-heap uses fir::replaceAllocas (MemoryUtils);
//   - heap-to-stack reuses the StackArrays analysis and rewrite, which only
//     stackifies fir.allocmem that are provably freed on all paths.
//
//===----------------------------------------------------------------------===//

#include "StackArrays.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Transforms/AllocationPlacementPolicy.h"
#include "flang/Optimizer/Transforms/MemoryUtils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <optional>

namespace fir {
#define GEN_PASS_DEF_ALLOCATIONPLACEMENT
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "allocation-placement"

//===----------------------------------------------------------------------===//
// Default placement policy
//===----------------------------------------------------------------------===//

fir::AllocationPlacement
fir::decideAllocationPlacement(const AllocationInfo &info,
                               const AllocationPlacementThresholds &thresholds,
                               std::size_t stackBytesUsed) {
  using P = fir::AllocationPlacement;

  // Translate a "should this be on the stack" decision into a placement,
  // accounting for where the allocation currently lives.
  auto place = [&](bool wantStack) -> P {
    if (wantStack)
      return info.isCurrentlyOnStack ? P::Leave : P::Stack;
    return info.isCurrentlyOnStack ? P::Heap : P::Leave;
  };

  // -fstack-arrays: put everything on the stack (best effort). The
  // heap-to-stack conversion still only happens where it is provably safe.
  if (thresholds.stackArrays)
    return place(/*wantStack=*/true);

  // Runtime-sized arrays (automatic arrays, dynamic temporaries) go on the
  // heap.
  if (info.isDynamic)
    return place(/*wantStack=*/false);

  // Without a known constant size we cannot reason about thresholds.
  if (!info.byteSize)
    return P::Leave;

  // Constant-size user variables always go on the stack.
  if (!info.isTemporary)
    return place(/*wantStack=*/true);

  auto size = static_cast<std::size_t>(*info.byteSize);
  if (size <= thresholds.smallArrayThresholdBytes)
    // Small arrays go on the stack while the per-function budget allows it.
    return place(/*wantStack=*/stackBytesUsed + size <=
                 thresholds.totalStackLimitBytes);

  // Big array temporaries go on the heap.
  return place(/*wantStack=*/false);
}

namespace {

/// Return true if the allocation is a compiler temporary, i.e. it has no
/// uniqued name (user variables always carry one).
static bool isTemporaryAllocation(mlir::Operation *op) {
  std::optional<llvm::StringRef> uniqName;
  if (auto alloca = mlir::dyn_cast<fir::AllocaOp>(op))
    uniqName = alloca.getUniqName();
  else if (auto allocmem = mlir::dyn_cast<fir::AllocMemOp>(op))
    uniqName = allocmem.getUniqName();
  return !uniqName || uniqName->empty();
}

/// Return the constant byte size of an array allocation, or std::nullopt if it
/// is dynamic or cannot be determined.
static std::optional<std::int64_t>
getConstantByteSize(mlir::Operation *op,
                    const std::optional<mlir::DataLayout> &dl,
                    const std::optional<fir::KindMapping> &kindMap) {
  if (!dl || !kindMap)
    return std::nullopt;
  if (auto alloca = mlir::dyn_cast<fir::AllocaOp>(op))
    return fir::getAllocaByteSize(alloca, *dl, *kindMap);
  if (auto allocmem = mlir::dyn_cast<fir::AllocMemOp>(op)) {
    if (allocmem.hasLenParams() || allocmem.hasShapeOperands())
      return std::nullopt;
    if (auto sizeAndAlignment = fir::getTypeSizeAndAlignment(
            op->getLoc(), allocmem.getAllocatedType(), *dl, *kindMap))
      return static_cast<std::int64_t>(sizeAndAlignment->first);
  }
  return std::nullopt;
}

/// Replacement generator used for stack-to-heap conversions (fir.alloca ->
/// fir.allocmem). Mirrors the MemoryAllocation pass.
static mlir::Value genAllocmem(mlir::OpBuilder &builder, fir::AllocaOp alloca,
                               bool /*deallocPointsDominateAlloc*/) {
  mlir::Type varTy = alloca.getInType();
  auto unpackName = [](std::optional<llvm::StringRef> opt) -> llvm::StringRef {
    if (opt)
      return *opt;
    return {};
  };
  llvm::StringRef uniqName = unpackName(alloca.getUniqName());
  llvm::StringRef bindcName = unpackName(alloca.getBindcName());
  auto heap = fir::AllocMemOp::create(builder, alloca.getLoc(), varTy, uniqName,
                                      bindcName, alloca.getTypeparams(),
                                      alloca.getShape());
  LLVM_DEBUG(llvm::dbgs() << "allocation placement: replaced " << alloca
                          << " with " << heap << '\n');
  return heap;
}

static void genFreemem(mlir::Location loc, mlir::OpBuilder &builder,
                       mlir::Value allocmem) {
  [[maybe_unused]] auto free = fir::FreeMemOp::create(builder, loc, allocmem);
  LLVM_DEBUG(llvm::dbgs() << "allocation placement: add free " << free
                          << " for " << allocmem << '\n');
}

[[maybe_unused]] static llvm::StringRef
placementName(fir::AllocationPlacement placement) {
  switch (placement) {
  case fir::AllocationPlacement::Stack:
    return "stack";
  case fir::AllocationPlacement::Heap:
    return "heap";
  case fir::AllocationPlacement::Leave:
    return "leave";
  }
  return "?";
}

/// Return true if \p placement leaves the allocation on the stack.
static bool endsUpOnStack(fir::AllocationPlacement placement,
                          bool isCurrentlyOnStack) {
  return placement == fir::AllocationPlacement::Stack ||
         (placement == fir::AllocationPlacement::Leave && isCurrentlyOnStack);
}

class AllocationPlacementPass
    : public fir::impl::AllocationPlacementBase<AllocationPlacementPass> {
public:
  AllocationPlacementPass() = default;
  AllocationPlacementPass(const AllocationPlacementPass &pass)
      : fir::impl::AllocationPlacementBase<AllocationPlacementPass>(pass),
        placementHook(pass.placementHook) {}
  AllocationPlacementPass(fir::AllocationPlacementOptions options)
      : fir::impl::AllocationPlacementBase<AllocationPlacementPass>(
            std::move(options)) {}
  AllocationPlacementPass(fir::AllocationPlacementOptions options,
                          fir::AllocationPlacementHook hook)
      : fir::impl::AllocationPlacementBase<AllocationPlacementPass>(
            std::move(options)),
        placementHook(std::move(hook)) {}

  void runOnOperation() override;

private:
  fir::AllocationPlacementHook placementHook;
};

void AllocationPlacementPass::runOnOperation() {
  mlir::func::FuncOp func = getOperation();
  if (func.empty())
    return;

  fir::AllocationPlacementThresholds baseThresholds;
  baseThresholds.stackArrays = stackArrays;
  baseThresholds.smallArrayThresholdBytes = smallArrayThresholdBytes;
  baseThresholds.totalStackLimitBytes = totalStackLimitBytes;

  auto module = func->getParentOfType<mlir::ModuleOp>();
  std::optional<mlir::DataLayout> dl =
      module ? fir::support::getOrSetMLIRDataLayout(
                   module, /*allowDefaultLayout=*/false)
             : std::nullopt;
  std::optional<fir::KindMapping> kindMap;
  if (module)
    kindMap = fir::getKindMapping(module);

  // Analysis of which fir.allocmem can be safely moved to the stack, and where.
  auto &analysis = getAnalysis<fir::StackArraysAnalysisWrapper>();
  const fir::StackArraysAnalysisWrapper::AllocMemMap *candidateOps =
      analysis.getCandidateOps(func);
  if (!candidateOps) {
    signalPassFailure();
    return;
  }

  // Walk allocations in deterministic program order, maintaining the running
  // per-function stack budget while collecting the conversions to perform.
  std::size_t stackBytesUsed = 0;
  llvm::DenseSet<mlir::Operation *> allocasToHeap;
  llvm::SmallVector<mlir::Operation *> allocmemsToStack;

  func.walk([&](mlir::Operation *op) {
    auto alloca = mlir::dyn_cast<fir::AllocaOp>(op);
    auto allocmem = mlir::dyn_cast<fir::AllocMemOp>(op);
    if (!alloca && !allocmem)
      return;

    // Only array allocations are considered.
    mlir::Type inTy = alloca ? alloca.getInType() : allocmem.getAllocatedType();
    if (!mlir::isa<fir::SequenceType>(inTy))
      return;

    fir::AllocationInfo info;
    info.op = op;
    info.isCurrentlyOnStack = static_cast<bool>(alloca);
    info.isTemporary = isTemporaryAllocation(op);
    info.isDynamic =
        alloca ? alloca.isDynamic()
               : (allocmem.hasLenParams() || allocmem.hasShapeOperands());
    info.byteSize = getConstantByteSize(op, dl, kindMap);

    // A hook, if provided, fully overrides the default policy; it may delegate
    // back to decideAllocationPlacement after adjusting the thresholds.
    fir::AllocationPlacement placement =
        placementHook ? placementHook(info, baseThresholds, stackBytesUsed)
                      : fir::decideAllocationPlacement(info, baseThresholds,
                                                       stackBytesUsed);

    // Account for the decision in the running stack budget.
    if (endsUpOnStack(placement, info.isCurrentlyOnStack) && info.byteSize)
      stackBytesUsed += static_cast<std::size_t>(*info.byteSize);

    LLVM_DEBUG({
      llvm::dbgs() << "allocation-placement: "
                   << (info.isCurrentlyOnStack ? "alloca" : "allocmem") << " "
                   << (info.isTemporary ? "temp" : "user") << " ";
      if (info.isDynamic)
        llvm::dbgs() << "dynamic";
      else if (info.byteSize)
        llvm::dbgs() << *info.byteSize << "B";
      else
        llvm::dbgs() << "unknown-size";
      llvm::dbgs() << " -> " << placementName(placement) << "\n";
    });

    if (alloca && placement == fir::AllocationPlacement::Heap)
      allocasToHeap.insert(op);
    else if (allocmem && placement == fir::AllocationPlacement::Stack &&
             candidateOps->contains(op))
      allocmemsToStack.push_back(op);
  });

  LLVM_DEBUG(llvm::dbgs() << "allocation-placement: " << func.getSymName()
                          << ": " << allocasToHeap.size() << " stack->heap, "
                          << allocmemsToStack.size() << " heap->stack, "
                          << stackBytesUsed << " stack bytes used\n");

  // Heap-to-stack: only provably-safe candidates are converted.
  if (!allocmemsToStack.empty()) {
    mlir::MLIRContext &context = getContext();
    mlir::RewritePatternSet patterns(&context);
    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks; otherwise use the default
    // configuration (folding on) as the legacy stack-arrays pass did.
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);
    patterns.insert<fir::AllocMemConversion>(&context, *candidateOps, dl,
                                             kindMap);
    if (mlir::failed(mlir::applyOpPatternsGreedily(
            allocmemsToStack, std::move(patterns), config))) {
      mlir::emitError(func->getLoc(),
                      "error in allocation placement (heap to stack)\n");
      signalPassFailure();
      return;
    }
  }

  // Stack-to-heap.
  if (!allocasToHeap.empty()) {
    mlir::IRRewriter rewriter(&getContext());
    auto mustReplace = [&](fir::AllocaOp alloca) {
      return allocasToHeap.contains(alloca.getOperation());
    };
    fir::replaceAllocas(rewriter, func.getOperation(), mustReplace, genAllocmem,
                        genFreemem);
  }
}

} // namespace

std::unique_ptr<mlir::Pass>
fir::createAllocationPlacement(const fir::AllocationPlacementOptions &options,
                               fir::AllocationPlacementHook placementHook) {
  return std::make_unique<AllocationPlacementPass>(options,
                                                   std::move(placementHook));
}
