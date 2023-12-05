//===- OpenACCLoopExpand.cpp - expand acc.loop operand to fir.do_loop nest ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace fir {
#define GEN_PASS_DEF_OPENACCLOOPEXPAND
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

class LoopExpand : public fir::impl::OpenACCLoopExpandBase<LoopExpand> {
public:
  void runOnOperation() override;
};

static mlir::Value retrievePrivatizedIv(mlir::acc::LoopOp &op,
                                        mlir::Value value) {
  for (auto p : op.getPrivateOperands()) {
    if (p == value) {
      auto privateOp = mlir::cast<mlir::acc::PrivateOp>(p.getDefiningOp());
      return privateOp.getVarPtr();
    }
  }
  return mlir::Value{};
}

/// Reset operands and operand segments for the induction ranges.
static void clearInductionRangesAndAttrs(fir::FirOpBuilder &builder,
                                         mlir::acc::LoopOp &accLoopOp) {
  // Remove the ranges.
  accLoopOp.getLowerboundMutable().clear();
  accLoopOp.getUpperboundMutable().clear();
  accLoopOp.getStepMutable().clear();

  accLoopOp.removeInclusiveUpperboundAttr();
}

static llvm::SmallVector<mlir::Value>
getOriginalInductionVars(mlir::acc::LoopOp &accLoopOp) {
  llvm::SmallVector<mlir::Value> ivs;
  for (auto arg : accLoopOp.getBody().getArguments()) {
    mlir::Value privateValue;
    for (mlir::OpOperand &u : arg.getUses()) {
      mlir::Operation *owner = u.getOwner();
      if (auto storeOp = mlir::dyn_cast<fir::StoreOp>(owner)) {
        privateValue = storeOp.getMemref();
        owner->erase();
      }
    }
    mlir::Value originalIv = retrievePrivatizedIv(accLoopOp, privateValue);
    assert(originalIv && "Expect induction variable to be found");
    ivs.push_back(originalIv);
  }
  return ivs;
}

static unsigned getOperandPosition(const mlir::OperandRange &operands,
                                   mlir::Value val) {
  unsigned pos = 0;
  for (auto v : operands) {
    if (v == val)
      return pos;
    ++pos;
  }
  return UINT_MAX;
}

static void clearIVPrivatizations(llvm::SmallVector<mlir::Value> &ivs,
                                  mlir::acc::LoopOp &accLoopOp,
                                  mlir::MLIRContext *context) {
  // Collect all of the private operations associated with IVs.
  llvm::SmallVector<mlir::Value> privOps;
  for (auto priv : accLoopOp.getPrivateOperands()) {
    mlir::Value varPtr{mlir::acc::getVarPtr(priv.getDefiningOp())};
    assert(varPtr && "must be able to extract varPtr from acc private op");
    if (llvm::find(ivs, varPtr) != ivs.end()) {
      privOps.push_back(priv);
    }
  }

  // Next remove the private operations associated with IVs.
  for (auto priv : privOps) {
    llvm::errs() << priv << "\n";
    mlir::Value varPtr{mlir::acc::getVarPtr(priv.getDefiningOp())};
    auto pos = getOperandPosition(accLoopOp.getPrivateOperands(), priv);

    // 1) Replace all uses of the private var with the varPtr.
    priv.replaceAllUsesWith(varPtr);

    // 2) Manually handle the loop operation since the operand list containing
    accLoopOp.getPrivateOperandsMutable().erase(pos);
    std::vector<mlir::Attribute> updatedRecipesList;
    for (auto [index, attribute] :
         llvm::enumerate(accLoopOp.getPrivatizationsAttr())) {
      if (index != pos) {
        updatedRecipesList.push_back(attribute);
      }
    }
    if (updatedRecipesList.empty()) {
      accLoopOp.removePrivatizationsAttr();
    } else {
      accLoopOp.setPrivatizationsAttr(
          mlir::ArrayAttr::get(context, updatedRecipesList));
    }

    // 3) Now remove the private op.
    priv.getDefiningOp()->erase();
  }
}

void LoopExpand::runOnOperation() {
  mlir::func::FuncOp func = getOperation();

  mlir::ModuleOp mod = func->getParentOfType<mlir::ModuleOp>();
  fir::KindMapping kindMap = fir::getKindMapping(mod);
  fir::FirOpBuilder builder{mod, std::move(kindMap)};

  func.walk([&](mlir::acc::LoopOp accLoopOp) {
    mlir::Location loc = accLoopOp.getLoc();
    mlir::Type idxTy = builder.getIndexType();

    bool isStructured = accLoopOp.getLoopRegions().front()->hasOneBlock();
    bool finalCountValue = isStructured;
    unsigned nbLoop = accLoopOp.getBody().getNumArguments();

    // Gather original (non-privatized) induction variables.
    llvm::SmallVector<mlir::Value> ivs = getOriginalInductionVars(accLoopOp);

    // Clear the privatization list from privatized IVs.
    clearIVPrivatizations(ivs, accLoopOp, &getContext());

    // Remove block arguments in order to create loop-nest and move current body
    // in the newly created loop nest.
    accLoopOp.getBody().eraseArguments(0, nbLoop);
    builder.setInsertionPointAfter(accLoopOp);

    if (!isStructured) {
      clearInductionRangesAndAttrs(builder, accLoopOp);
      return;
    }

    llvm::SmallVector<mlir::Value> lbs, ubs, steps, iterArgs;
    llvm::SmallVector<fir::DoLoopOp> loops;

    // Create the loop nest, move the acc.loop body inside and move the loop
    // nest inside the acc.loop again.
    for (unsigned i = 0; i < nbLoop; ++i) {
      bool isInnerLoop = i == (nbLoop - 1);

      lbs.push_back(
          builder.createConvert(loc, idxTy, accLoopOp.getLowerbound()[i]));
      ubs.push_back(
          builder.createConvert(loc, idxTy, accLoopOp.getUpperbound()[i]));
      steps.push_back(
          builder.createConvert(loc, idxTy, accLoopOp.getStep()[i]));
      iterArgs.push_back(builder.createConvert(loc, fir::unwrapRefType(ivs[i].getType()), accLoopOp.getLowerbound()[i]));
      fir::DoLoopOp doLoopOp = builder.create<fir::DoLoopOp>(
          loc, lbs[i], ubs[i], steps[i], /*unordered=*/false, finalCountValue,
          mlir::ValueRange{iterArgs[i]});
      loops.push_back(doLoopOp);

      if (isInnerLoop) {
        // Move acc.loop body inside the newly created fir.do_loop.
        accLoopOp.getBody().getTerminator()->erase();
        doLoopOp.getRegion().takeBody(*accLoopOp.getLoopRegions().front());
        // Recreate the block arguments.
        doLoopOp.getBody()->addArgument(builder.getIndexType(), loc);
        doLoopOp.getBody()->addArgument(iterArgs[i].getType(), loc);
      } else {
        builder.setInsertionPointToStart(doLoopOp.getBody());
      }
    }

    // Move the loop nest inside the acc.loop region.
    mlir::Block *newAccLoopBlock =
        builder.createBlock(accLoopOp.getLoopRegions().front());
    loops[0].getOperation()->moveBefore(newAccLoopBlock,
                                        newAccLoopBlock->end());

    for (unsigned i = 0; i < nbLoop; ++i) {
      builder.setInsertionPointToStart(loops[i].getBody());
      builder.create<fir::StoreOp>(loc, loops[i].getBody()->getArgument(1),
                                   ivs[i]);

      builder.setInsertionPointToEnd(loops[i].getBody());
      llvm::SmallVector<mlir::Value, 2> results;
      if (finalCountValue)
        results.push_back(builder.create<mlir::arith::AddIOp>(
            loc, loops[i].getInductionVar(), loops[i].getStep()));

      // Step loopVariable to help optimizations such as vectorization.
      // Induction variable elimination will clean up as necessary.
      mlir::Value loopVar = builder.create<fir::LoadOp>(loc, ivs[i]);
      mlir::Value convStep = builder.create<fir::ConvertOp>(
          loc, loopVar.getType(), loops[i].getStep());
      results.push_back(
          builder.create<mlir::arith::AddIOp>(loc, loopVar, convStep));
      builder.create<fir::ResultOp>(loc, results);

      // Convert ops have been created outside of the acc.loop operation. They
      // need to be moved back before their uses.
      if (mlir::isa<fir::ConvertOp>(lbs[i].getDefiningOp()))
        lbs[i].getDefiningOp()->moveBefore(loops[i].getOperation());
      if (mlir::isa<fir::ConvertOp>(ubs[i].getDefiningOp()))
        ubs[i].getDefiningOp()->moveBefore(loops[i].getOperation());
      if (mlir::isa<fir::ConvertOp>(steps[i].getDefiningOp()))
        steps[i].getDefiningOp()->moveBefore(loops[i].getOperation());
      if (mlir::isa<fir::ConvertOp>(iterArgs[i].getDefiningOp()))
        iterArgs[i].getDefiningOp()->moveBefore(loops[i].getOperation());
    }

    builder.setInsertionPointToEnd(newAccLoopBlock);
    builder.create<mlir::acc::YieldOp>(loc);

    clearInductionRangesAndAttrs(builder, accLoopOp);
  });
}

std::unique_ptr<mlir::Pass> fir::createOpenACCLoopExpandPass() {
  return std::make_unique<LoopExpand>();
}
