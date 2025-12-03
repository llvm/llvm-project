//===- ACCRecipeBufferization.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bufferize OpenACC recipes that yield fir.box<T> to operate on
// fir.ref<fir.box<T>> and update uses accordingly.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/OpenACC/Passes.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/TypeSwitch.h"

namespace fir::acc {
#define GEN_PASS_DEF_ACCRECIPEBUFFERIZATION
#include "flang/Optimizer/OpenACC/Passes.h.inc"
} // namespace fir::acc

namespace {

class BufferizeInterface {
public:
  static std::optional<mlir::Type> mustBufferize(mlir::Type recipeType) {
    if (auto boxTy = llvm::dyn_cast<fir::BaseBoxType>(recipeType))
      return fir::ReferenceType::get(boxTy);
    return std::nullopt;
  }

  static mlir::Operation *load(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value value) {
    return fir::LoadOp::create(builder, loc, value);
  }

  static mlir::Value placeInMemory(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value value) {
    auto alloca = fir::AllocaOp::create(builder, loc, value.getType());
    fir::StoreOp::create(builder, loc, value, alloca);
    return alloca;
  }
};

static void bufferizeRegionArgsAndYields(mlir::Region &region,
                                         mlir::Location loc, mlir::Type oldType,
                                         mlir::Type newType) {
  if (region.empty())
    return;

  mlir::OpBuilder builder(&region);
  for (mlir::BlockArgument arg : region.getArguments()) {
    if (arg.getType() == oldType) {
      arg.setType(newType);
      if (!arg.use_empty()) {
        mlir::Operation *loadOp = BufferizeInterface::load(builder, loc, arg);
        arg.replaceAllUsesExcept(loadOp->getResult(0), loadOp);
      }
    }
  }
  if (auto yield =
          llvm::dyn_cast<mlir::acc::YieldOp>(region.back().getTerminator())) {
    llvm::SmallVector<mlir::Value> newOperands;
    newOperands.reserve(yield.getNumOperands());
    bool changed = false;
    for (mlir::Value oldYieldArg : yield.getOperands()) {
      if (oldYieldArg.getType() == oldType) {
        builder.setInsertionPoint(yield);
        mlir::Value alloca =
            BufferizeInterface::placeInMemory(builder, loc, oldYieldArg);
        newOperands.push_back(alloca);
        changed = true;
      } else {
        newOperands.push_back(oldYieldArg);
      }
    }
    if (changed)
      yield->setOperands(newOperands);
  }
}

template <typename OpTy>
static void updateRecipeUse(mlir::ValueRange operands,
                            llvm::StringRef recipeSymName,
                            mlir::Operation *computeOp) {
  for (auto operand : operands) {
    auto op = operand.getDefiningOp<OpTy>();
    if (!op || !op.getRecipe().has_value() ||
        op.getRecipeAttr().getLeafReference() != recipeSymName)
      continue;

    mlir::Location loc = op->getLoc();

    mlir::OpBuilder builder(op);
    builder.setInsertionPointAfterValue(op.getVar());
    mlir::Value alloca =
        BufferizeInterface::placeInMemory(builder, loc, op.getVar());
    op.getVarMutable().assign(alloca);
    op.getAccVar().setType(alloca.getType());

    mlir::Value oldRes = op.getAccVar();
    llvm::SmallVector<mlir::Operation *> users(oldRes.getUsers().begin(),
                                               oldRes.getUsers().end());
    for (mlir::Operation *useOp : users) {
      if (useOp == computeOp)
        continue;
      builder.setInsertionPoint(useOp);
      mlir::Operation *load = BufferizeInterface::load(builder, loc, oldRes);
      useOp->replaceUsesOfWith(oldRes, load->getResult(0));
    }
  }
}

class ACCRecipeBufferization
    : public fir::acc::impl::ACCRecipeBufferizationBase<
          ACCRecipeBufferization> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<llvm::StringRef> recipeNames;
    module.walk([&](mlir::Operation *recipe) {
      llvm::TypeSwitch<mlir::Operation *, void>(recipe)
          .Case<mlir::acc::PrivateRecipeOp, mlir::acc::FirstprivateRecipeOp,
                mlir::acc::ReductionRecipeOp>([&](auto recipe) {
            mlir::Type oldType = recipe.getType();
            auto bufferizedType =
                BufferizeInterface::mustBufferize(recipe.getType());
            if (!bufferizedType)
              return;
            recipe.setTypeAttr(mlir::TypeAttr::get(*bufferizedType));
            mlir::Location loc = recipe.getLoc();
            using RecipeOp = decltype(recipe);
            bufferizeRegionArgsAndYields(recipe.getInitRegion(), loc, oldType,
                                         *bufferizedType);
            if constexpr (std::is_same_v<RecipeOp,
                                         mlir::acc::FirstprivateRecipeOp>)
              bufferizeRegionArgsAndYields(recipe.getCopyRegion(), loc, oldType,
                                           *bufferizedType);
            if constexpr (std::is_same_v<RecipeOp,
                                         mlir::acc::ReductionRecipeOp>)
              bufferizeRegionArgsAndYields(recipe.getCombinerRegion(), loc,
                                           oldType, *bufferizedType);
            bufferizeRegionArgsAndYields(recipe.getDestroyRegion(), loc,
                                         oldType, *bufferizedType);
            recipeNames.push_back(recipe.getSymName());
          });
    });
    if (recipeNames.empty())
      return;

    module.walk([&](mlir::Operation *op) {
      llvm::TypeSwitch<mlir::Operation *, void>(op)
          .Case<mlir::acc::LoopOp, mlir::acc::ParallelOp, mlir::acc::SerialOp>(
              [&](auto computeOp) {
                for (llvm::StringRef recipeName : recipeNames) {
                  if (!computeOp.getPrivateOperands().empty())
                    updateRecipeUse<mlir::acc::PrivateOp>(
                        computeOp.getPrivateOperands(), recipeName, op);
                  if (!computeOp.getFirstprivateOperands().empty())
                    updateRecipeUse<mlir::acc::FirstprivateOp>(
                        computeOp.getFirstprivateOperands(), recipeName, op);
                  if (!computeOp.getReductionOperands().empty())
                    updateRecipeUse<mlir::acc::ReductionOp>(
                        computeOp.getReductionOperands(), recipeName, op);
                }
              });
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> fir::acc::createACCRecipeBufferizationPass() {
  return std::make_unique<ACCRecipeBufferization>();
}
