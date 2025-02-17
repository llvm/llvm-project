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

static bool isOpInLoop(mlir::Operation *op) {
  return op->getParentOfType<cir::LoopOpInterface>();
}

static bool hasStoreToAllocaInWhileCond(cir::AllocaOp alloca) {
  // This function determines whether the given alloca operation represents
  // a variable defined as a while loop's condition.
  //
  // Specifically, C/C++ allows the condition of a while loop be a variable
  // declaration:
  //
  //   while (const int x = foo()) { /* body... */ }
  //
  // CIRGen would emit the following CIR for the above code:
  //
  //   cir.scope {
  //     %x.slot = cir.alloca !s32i [init, const]
  //     cir.while {
  //       %0 = cir.call @foo()
  //       cir.store %0, %x
  //       %1 = cir.load %x
  //       %2 = cir.cast int_to_bool %1
  //       cir.condition(%2)
  //     } do {
  //       // loop body goes here.
  //     }
  //   }
  //
  // Note that %x.slot is emitted outside the cir.while operation. Ideally, the
  // cir.while operation should cover this cir.alloca operation, but currently
  // CIR does not work this way. When hoisting such an alloca operation, one
  // must remove the "const" flag from it, otherwise LLVM lowering code will
  // mistakenly attach invariant group metadata to the load and store operations
  // in the while body, indicating that all loads and stores across all
  // iterations of the loop are constant.

  for (mlir::Operation *user : alloca->getUsers()) {
    if (!mlir::isa<cir::StoreOp>(user))
      continue;

    auto store = mlir::cast<cir::StoreOp>(user);
    mlir::Operation *storeParentOp = store->getParentOp();
    if (!mlir::isa<cir::WhileOp>(storeParentOp))
      continue;

    auto whileOp = mlir::cast<cir::WhileOp>(storeParentOp);
    return &whileOp.getCond() == store->getParentRegion();
  }

  return false;
}

static void processConstAlloca(cir::AllocaOp alloca) {
  // When optimization is enabled, LLVM lowering would start emitting invariant
  // group metadata for loads and stores to alloca-ed objects with "const"
  // attribute. For example, the following CIR:
  //
  //   %slot = cir.alloca !s32i [init, const]
  //   cir.store %0, %slot
  //   %1 = cir.load %slot
  //
  // would be lowered to the following LLVM IR:
  //
  //   %slot = alloca i32, i64 1
  //   store i32 %0, ptr %slot, !invariant.group !0
  //   %1 = load i32, ptr %slot, !invariant.group !0
  //
  // The invariant group metadata would tell LLVM optimizer that the store and
  // load instruction would store and load the same value from %slot.
  //
  // So far so good. Things started to get tricky when such an alloca operation
  // appears in the body of a loop construct:
  //
  //   cir.some_loop_construct {
  //     %slot = cir.alloca !s32i [init, const]
  //     cir.store %0, %slot
  //     %1 = cir.load %slot
  //   }
  //
  // After alloca hoisting, the CIR code above would be transformed into:
  //
  //   %slot = cir.alloca !s32i [init, const]
  //   cir.some_loop_construct {
  //     cir.store %0, %slot
  //     %1 = cir.load %slot
  //   }
  //
  // Notice how alloca hoisting change the semantics of the program in such a
  // case. The transformed code now indicates the optimizer that the load and
  // store operations load and store the same value **across all iterations of
  // the loop**!
  //
  // To overcome this problem, we instead transform the program into this:
  //
  //   %slot = cir.alloca !s32i [init, const]
  //   cir.some_loop_construct {
  //     %slot.inv = cir.invariant_group %slot
  //     cir.store %0, %slot.inv
  //     %1 = cir.load %slot.inv
  //   }
  //
  // The cir.invariant_group operation attaches fresh invariant information to
  // the operand pointer and yields a pointer with the fresh invariant
  // information. Upon each loop iteration, the old invariant information is
  // disgarded, and a new invariant information is attached, thus the correct
  // program semantic retains. During LLVM lowering, the cir.invariant_group
  // operation would eventually become an intrinsic call to
  // @llvm.launder.invariant.group.

  if (isOpInLoop(alloca)) {
    // Mark the alloca-ed pointer as invariant via the cir.invariant_group
    // operation.
    mlir::OpBuilder builder(alloca);
    auto invariantGroupOp =
        builder.create<cir::InvariantGroupOp>(alloca.getLoc(), alloca);

    // And replace all uses of the original alloca-ed pointer with the marked
    // pointer (which carries invariant group information).
    alloca->replaceUsesWithIf(
        invariantGroupOp,
        [op = invariantGroupOp.getOperation()](mlir::OpOperand &use) {
          return use.getOwner() != op;
        });
  } else if (hasStoreToAllocaInWhileCond(alloca)) {
    // The alloca represents a variable declared as the condition of a while
    // loop. In CIR, the alloca would be emitted at a scope outside of the
    // while loop. We have to remove the constant flag during hoisting,
    // otherwise we would be telling the optimizer that the alloca-ed value
    // is constant across all iterations of the while loop.
    //
    // See the body of the isWhileCondition function for more details.
    alloca.setConstant(false);
  }
}

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
    if (alloca.getDynAllocSize())
      return;
    allocas.push_back(alloca);
  });
  if (allocas.empty())
    return;

  mlir::Operation *insertPoint = &*entryBlock.begin();
  auto optInfoAttr = mlir::cast_if_present<cir::OptInfoAttr>(
      mod->getAttr(cir::CIRDialect::getOptInfoAttrName()));
  unsigned optLevel = optInfoAttr ? optInfoAttr.getLevel() : 0;

  for (auto alloca : allocas) {
    if (alloca.getConstant()) {
      if (optLevel == 0) {
        // Under non-optimized builds, just remove the constant flag.
        alloca.setConstant(false);
        continue;
      }

      processConstAlloca(alloca);
    }

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