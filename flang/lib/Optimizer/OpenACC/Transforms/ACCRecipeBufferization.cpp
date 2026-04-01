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
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/IR/Block.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/IR/Value.h"
#include "aiir/IR/Visitors.h"
#include "llvm/ADT/TypeSwitch.h"

namespace fir::acc {
#define GEN_PASS_DEF_ACCRECIPEBUFFERIZATION
#include "flang/Optimizer/OpenACC/Passes.h.inc"
} // namespace fir::acc

namespace {

class BufferizeInterface {
public:
  static std::optional<aiir::Type> mustBufferize(aiir::Type recipeType) {
    if (auto boxTy = llvm::dyn_cast<fir::BaseBoxType>(recipeType))
      return fir::ReferenceType::get(boxTy);
    return std::nullopt;
  }

  static aiir::Operation *load(aiir::OpBuilder &builder, aiir::Location loc,
                               aiir::Value value) {
    return fir::LoadOp::create(builder, loc, value);
  }

  static aiir::Value placeInMemory(aiir::OpBuilder &builder, aiir::Location loc,
                                   aiir::Value value) {
    auto alloca = fir::AllocaOp::create(builder, loc, value.getType());
    fir::StoreOp::create(builder, loc, value, alloca);
    return alloca;
  }
};

static void bufferizeRegionArgsAndYields(aiir::Region &region,
                                         aiir::Location loc, aiir::Type oldType,
                                         aiir::Type newType) {
  if (region.empty())
    return;

  aiir::OpBuilder builder(&region);
  for (aiir::BlockArgument arg : region.getArguments()) {
    if (arg.getType() == oldType) {
      arg.setType(newType);
      if (!arg.use_empty()) {
        aiir::Operation *loadOp = BufferizeInterface::load(builder, loc, arg);
        arg.replaceAllUsesExcept(loadOp->getResult(0), loadOp);
      }
    }
  }
  if (auto yield =
          llvm::dyn_cast<aiir::acc::YieldOp>(region.back().getTerminator())) {
    llvm::SmallVector<aiir::Value> newOperands;
    newOperands.reserve(yield.getNumOperands());
    bool changed = false;
    for (aiir::Value oldYieldArg : yield.getOperands()) {
      if (oldYieldArg.getType() == oldType) {
        builder.setInsertionPoint(yield);
        aiir::Value alloca =
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
static void updateRecipeUse(aiir::ValueRange operands,
                            llvm::StringRef recipeSymName,
                            aiir::Operation *computeOp) {
  for (auto operand : operands) {
    auto op = operand.getDefiningOp<OpTy>();
    if (!op || !op.getRecipe().has_value() ||
        op.getRecipeAttr().getLeafReference() != recipeSymName)
      continue;

    aiir::Location loc = op->getLoc();

    aiir::OpBuilder builder(op);
    builder.setInsertionPointAfterValue(op.getVar());
    aiir::Value alloca =
        BufferizeInterface::placeInMemory(builder, loc, op.getVar());
    op.getVarMutable().assign(alloca);
    op.getAccVar().setType(alloca.getType());

    aiir::Value oldRes = op.getAccVar();
    llvm::SmallVector<aiir::Operation *> users(oldRes.getUsers().begin(),
                                               oldRes.getUsers().end());
    for (aiir::Operation *useOp : users) {
      if (useOp == computeOp)
        continue;
      builder.setInsertionPoint(useOp);
      aiir::Operation *load = BufferizeInterface::load(builder, loc, oldRes);
      useOp->replaceUsesOfWith(oldRes, load->getResult(0));
    }
  }
}

class ACCRecipeBufferization
    : public fir::acc::impl::ACCRecipeBufferizationBase<
          ACCRecipeBufferization> {
public:
  void runOnOperation() override {
    aiir::ModuleOp module = getOperation();

    llvm::SmallVector<llvm::StringRef> recipeNames;
    module.walk([&](aiir::Operation *recipe) {
      llvm::TypeSwitch<aiir::Operation *, void>(recipe)
          .Case<aiir::acc::PrivateRecipeOp, aiir::acc::FirstprivateRecipeOp,
                aiir::acc::ReductionRecipeOp>([&](auto recipe) {
            aiir::Type oldType = recipe.getType();
            auto bufferizedType =
                BufferizeInterface::mustBufferize(recipe.getType());
            if (!bufferizedType)
              return;
            recipe.setTypeAttr(aiir::TypeAttr::get(*bufferizedType));
            aiir::Location loc = recipe.getLoc();
            using RecipeOp = decltype(recipe);
            bufferizeRegionArgsAndYields(recipe.getInitRegion(), loc, oldType,
                                         *bufferizedType);
            if constexpr (std::is_same_v<RecipeOp,
                                         aiir::acc::FirstprivateRecipeOp>)
              bufferizeRegionArgsAndYields(recipe.getCopyRegion(), loc, oldType,
                                           *bufferizedType);
            if constexpr (std::is_same_v<RecipeOp,
                                         aiir::acc::ReductionRecipeOp>)
              bufferizeRegionArgsAndYields(recipe.getCombinerRegion(), loc,
                                           oldType, *bufferizedType);
            bufferizeRegionArgsAndYields(recipe.getDestroyRegion(), loc,
                                         oldType, *bufferizedType);
            recipeNames.push_back(recipe.getSymName());
          });
    });
    if (recipeNames.empty())
      return;

    module.walk([&](aiir::Operation *op) {
      llvm::TypeSwitch<aiir::Operation *, void>(op)
          .Case<aiir::acc::LoopOp, aiir::acc::ParallelOp, aiir::acc::SerialOp>(
              [&](auto computeOp) {
                for (llvm::StringRef recipeName : recipeNames) {
                  if (!computeOp.getPrivateOperands().empty())
                    updateRecipeUse<aiir::acc::PrivateOp>(
                        computeOp.getPrivateOperands(), recipeName, op);
                  if (!computeOp.getFirstprivateOperands().empty())
                    updateRecipeUse<aiir::acc::FirstprivateOp>(
                        computeOp.getFirstprivateOperands(), recipeName, op);
                  if (!computeOp.getReductionOperands().empty())
                    updateRecipeUse<aiir::acc::ReductionOp>(
                        computeOp.getReductionOperands(), recipeName, op);
                }
              });
    });
  }
};

} // namespace

std::unique_ptr<aiir::Pass> fir::acc::createACCRecipeBufferizationPass() {
  return std::make_unique<ACCRecipeBufferization>();
}
