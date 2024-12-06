//====- HoistAllocas.cpp --------------------------------------------------===//
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

#include "llvm/Support/TimeProfiler.h"

using namespace mlir;
using namespace cir;

namespace {

struct HoistAllocasPass : public HoistAllocasBase<HoistAllocasPass> {

  HoistAllocasPass() = default;
  void runOnOperation() override;
};

static void process(cir::FuncOp func) {
  if (func.getRegion().empty())
    return;

  // Hoist all static allocas to the entry block.
  mlir::Block &entryBlock = func.getRegion().front();
  llvm::SmallVector<cir::AllocaOp> allocas;
  func.getBody().walk([&](cir::AllocaOp alloca) {
    if (alloca->getBlock() == &entryBlock)
      return;
    // Don't hoist allocas with dynamic alloca size.
    if (alloca.getDynAllocSize())
      return;
    allocas.push_back(alloca);
  });
  if (allocas.empty())
    return;

  mlir::Operation *insertPoint = &*entryBlock.begin();

  for (auto alloca : allocas) {
    alloca->moveBefore(insertPoint);
    if (alloca.getConstant()) {
      // Hoisted alloca may come from the body of a loop, in which case the
      // stack slot is re-used by multiple objects alive in different iterations
      // of the loop. In theory, each of these objects are still constant within
      // their lifetimes, but currently we're not emitting metadata to further
      // describe this. So for now let's behave conservatively and remove the
      // const flag on nested allocas when hoisting them.
      alloca.setConstant(false);
    }
  }
}

void HoistAllocasPass::runOnOperation() {
  llvm::TimeTraceScope scope("Hoist Allocas");
  llvm::SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](cir::FuncOp op) { process(op); });
}

} // namespace

std::unique_ptr<Pass> mlir::createHoistAllocasPass() {
  return std::make_unique<HoistAllocasPass>();
}