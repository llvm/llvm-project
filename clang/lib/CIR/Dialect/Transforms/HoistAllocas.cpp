//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/TimeProfiler.h"

using namespace mlir;
using namespace cir;

namespace {

struct HoistAllocasPass : public HoistAllocasBase<HoistAllocasPass> {

  HoistAllocasPass() = default;
  void runOnOperation() override;
};

static void process(mlir::ModuleOp mod, cir::FuncOp func) {
  if (func.getRegion().empty())
    return;

  // Hoist all static allocas to the entry block.
  mlir::Block &entryBlock = func.getRegion().front();
  llvm::SmallVector<cir::AllocaOp> allocas;
  func.getBody().walk([&](cir::AllocaOp alloca) {
    if (alloca->getBlock() == &entryBlock)
      return;
    // Don't hoist allocas with dynamic alloca size.
    assert(!cir::MissingFeatures::opAllocaDynAllocSize());
    allocas.push_back(alloca);
  });
  if (allocas.empty())
    return;

  mlir::Operation *insertPoint = &*entryBlock.begin();

  for (auto alloca : allocas) {
    // Preserving the `const` attribute on hoisted allocas can cause LLVM to
    // incorrectly introduce invariant group metadata in some circumstances.
    // The incubator performs some analysis to determine whether the attribute
    // can be preserved, but it only runs this analysis when optimizations are
    // enabled. Until we start tracking the optimization level, we can just
    // always remove the `const` attribute.
    assert(!cir::MissingFeatures::optInfoAttr());
    if (alloca.getConstant())
      alloca.setConstant(false);

    alloca->moveBefore(insertPoint);
  }
}

void HoistAllocasPass::runOnOperation() {
  llvm::TimeTraceScope scope("Hoist Allocas");
  llvm::SmallVector<Operation *, 16> ops;

  Operation *op = getOperation();
  auto mod = mlir::dyn_cast<mlir::ModuleOp>(op);
  if (!mod)
    mod = op->getParentOfType<mlir::ModuleOp>();

  getOperation()->walk([&](cir::FuncOp op) { process(mod, op); });
}

} // namespace

std::unique_ptr<Pass> mlir::createHoistAllocasPass() {
  return std::make_unique<HoistAllocasPass>();
}
