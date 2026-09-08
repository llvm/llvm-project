//===- CudaHeapAllocPromotion.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Transforms/MemoryUtils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace fir {
#define GEN_PASS_DEF_CUDAHEAPALLOCPROMOTION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "cuda-heap-alloc-promotion"

namespace {
class CudaHeapAllocPromotion
    : public fir::impl::CudaHeapAllocPromotionBase<CudaHeapAllocPromotion> {
public:
  using CudaHeapAllocPromotionBase<
      CudaHeapAllocPromotion>::CudaHeapAllocPromotionBase;

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    if (func.empty())
      return;
    mlir::IRRewriter rewriter(&getContext());
    fir::promoteDynamicVariableAllocasToCudaHeap(rewriter, func.getOperation(),
                                                 stackArrays);
  }
};
} // namespace
