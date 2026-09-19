//===- LoopIdiomRecognize.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Eliminate redundant minMax tracking in equality-mask
// location-search loops after HLFIR-to-FIR lowering and LICM.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

namespace fir {
#define GEN_PASS_DEF_LOOPIDIOMRECOGNIZE
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

/// Return the array base reached via array_coor, designate, or apply, or
/// nullptr.
static mlir::Value getBase(mlir::Value v) {
  while (v) {
    while (auto conv = v.getDefiningOp<fir::ConvertOp>())
      v = conv.getOperand();

    if (!v)
      break;

    mlir::Operation *def = v.getDefiningOp();
    if (!def)
      break;

    if (auto coor = mlir::dyn_cast<fir::ArrayCoorOp>(def))
      return coor.getMemref();
    if (auto desig = mlir::dyn_cast<hlfir::DesignateOp>(def))
      return desig.getMemref();
    if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(def))
      return apply.getExpr();

    if (auto load = mlir::dyn_cast<fir::LoadOp>(def)) {
      if (fir::isa_volatile_type(load.getMemref().getType()) ||
          fir::isa_volatile_type(load.getType()))
        break;
      v = load.getMemref();
      continue;
    }

    break;
  }
  return nullptr;
}

/// Return the array being searched, extracted from the minMax then-yield.
static mlir::Value getSearchArray(fir::IfOp ifOp, unsigned minMaxIdx) {
  mlir::Block &thenBlock = ifOp.getThenRegion().front();
  auto thenYield = mlir::cast<fir::ResultOp>(thenBlock.getTerminator());
  mlir::Value minMax = thenYield.getOperand(minMaxIdx);

  while (auto sel = minMax.getDefiningOp<mlir::arith::SelectOp>())
    minMax = sel.getTrueValue();

  return getBase(minMax);
}

/// Match arith.cmpi eq where one side loads from searchArray and the other is
/// loop-invariant.
static mlir::Value getFIREqualityMaskTarget(mlir::Value cond,
                                            mlir::Value searchArray,
                                            fir::DoLoopOp rootLoop) {
  while (auto conv = cond.getDefiningOp<fir::ConvertOp>())
    cond = conv.getOperand();

  auto cmpOp = cond.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmpOp || cmpOp.getPredicate() != mlir::arith::CmpIPredicate::eq)
    return nullptr;

  auto canonicalizeBase = [](mlir::Value v) -> mlir::Value {
    while (v) {
      mlir::Operation *def = v.getDefiningOp();
      if (!def)
        break;
      if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(def))
        v = decl.getMemref();
      else if (mlir::isa<fir::ConvertOp, hlfir::AsExprOp>(def))
        v = def->getOperand(0);
      else
        break;
    }
    return v;
  };

  mlir::Value canonSearch = canonicalizeBase(searchArray);
  mlir::Value lhsBase = canonicalizeBase(getBase(cmpOp.getLhs()));
  mlir::Value rhsBase = canonicalizeBase(getBase(cmpOp.getRhs()));

  auto isInvariant = [&](mlir::Value v) -> bool {
    if (mlir::Operation *def = v.getDefiningOp())
      return !rootLoop->isAncestor(def);
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      mlir::Operation *ownerOp = blockArg.getOwner()->getParentOp();
      return !rootLoop->isAncestor(ownerOp) &&
             (ownerOp != rootLoop.getOperation());
    }
    return true;
  };

  if (lhsBase && lhsBase == canonSearch && !rhsBase &&
      isInvariant(cmpOp.getRhs()))
    return cmpOp.getRhs();
  if (rhsBase && rhsBase == canonSearch && !lhsBase &&
      isInvariant(cmpOp.getLhs()))
    return cmpOp.getLhs();
  return nullptr;
}

/// Drop the redundant minMax iter arg from an equality-mask location-search
/// loop nest.
static bool transformEqualityMaskMinMaxLoop(fir::DoLoopOp rootLoop,
                                            fir::FirOpBuilder &builder) {
  unsigned numArgs = rootLoop.getInitArgs().size();
  if (numArgs < 3)
    return false;
  if (rootLoop.getFinalValue().value_or(false))
    return false;

  unsigned isFirstIdx = numArgs - 1;
  unsigned minMaxIdx = numArgs - 2;

  if (!rootLoop.getResult(minMaxIdx).use_empty())
    return false;

  // Last iter arg must be i1 initialised to true.
  if (!rootLoop.getInitArgs()[isFirstIdx].getType().isInteger(1))
    return false;
  auto isFirstConst = rootLoop.getInitArgs()[isFirstIdx]
                          .getDefiningOp<mlir::arith::ConstantOp>();
  if (!isFirstConst)
    return false;
  auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(isFirstConst.getValue());
  if (!boolAttr || !boolAttr.getValue())
    return false;

  // Skip if the direct parent is the outer loop of the same nest.
  mlir::Operation *parent = rootLoop->getParentOp();
  if (auto parentLoop = mlir::dyn_cast_or_null<fir::DoLoopOp>(parent)) {
    if (parentLoop.getInitArgs().size() == numArgs) {
      auto parentIsFirstConst = parentLoop.getInitArgs()[isFirstIdx]
                                    .getDefiningOp<mlir::arith::ConstantOp>();
      if (parentIsFirstConst) {
        auto parentBoolAttr =
            mlir::dyn_cast<mlir::BoolAttr>(parentIsFirstConst.getValue());
        if (parentBoolAttr && parentBoolAttr.getValue())
          return false;
      }
    }
  }

  // Collect the contiguous loop nest.
  llvm::SmallVector<fir::DoLoopOp> nest;
  fir::DoLoopOp curr = rootLoop;
  while (curr) {
    nest.push_back(curr);
    mlir::Block *body = curr.getBody();
    fir::DoLoopOp inner;
    for (auto &op : *body) {
      if (auto nested = mlir::dyn_cast<fir::DoLoopOp>(op)) {
        if (nested.getInitArgs().size() == numArgs) {
          inner = nested;
          break;
        }
      }
    }
    curr = inner;
  }

  fir::DoLoopOp innerLoop = nest.back();

  // Find the fir.if inside the innermost loop body.
  fir::IfOp ifOp;
  for (auto &op : *innerLoop.getBody()) {
    if ((ifOp = mlir::dyn_cast<fir::IfOp>(op)))
      break;
  }
  if (!ifOp || ifOp.getThenRegion().empty() || ifOp.getElseRegion().empty())
    return false;

  mlir::Block *thenBlock = &ifOp.getThenRegion().front();
  auto thenYield = mlir::dyn_cast<fir::ResultOp>(thenBlock->getTerminator());
  if (!thenYield || thenYield.getNumOperands() != numArgs)
    return false;

  // Then-branch must yield false at isFirstIdx.
  auto thenIsFirstConst =
      thenYield.getOperand(isFirstIdx).getDefiningOp<mlir::arith::ConstantOp>();
  if (!thenIsFirstConst)
    return false;
  auto thenBoolAttr =
      mlir::dyn_cast<mlir::BoolAttr>(thenIsFirstConst.getValue());
  if (!thenBoolAttr || thenBoolAttr.getValue())
    return false;

  mlir::Block *elseBlock = &ifOp.getElseRegion().front();
  auto elseYield = mlir::dyn_cast<fir::ResultOp>(elseBlock->getTerminator());
  if (!elseYield || elseYield.getNumOperands() != numArgs)
    return false;

  // Else-branch must pass all iter args through unchanged.
  mlir::ValueRange elseArgs = innerLoop.getRegionIterArgs();
  for (unsigned i = 0; i < numArgs; ++i)
    if (elseYield.getOperand(i) != elseArgs[i])
      return false;

  mlir::Value searchArray = getSearchArray(ifOp, minMaxIdx);
  if (!searchArray)
    return false;

  mlir::Value target =
      getFIREqualityMaskTarget(ifOp.getCondition(), searchArray, rootLoop);
  if (!target)
    return false;

  mlir::Location loc = rootLoop.getLoc();

  // Drop minMax from then/else yields; unwrap arith.select for location args.
  builder.setInsertionPoint(thenYield);
  llvm::SmallVector<mlir::Value> newThenResults;
  for (unsigned i = 0; i < numArgs; ++i) {
    if (i == minMaxIdx)
      continue;
    if (i == isFirstIdx) {
      newThenResults.push_back(thenYield.getOperand(isFirstIdx));
    } else {
      mlir::Value val = thenYield.getOperand(i);
      while (auto sel = val.getDefiningOp<mlir::arith::SelectOp>())
        val = sel.getTrueValue();
      newThenResults.push_back(val);
    }
  }
  fir::ResultOp::create(builder, loc, newThenResults);
  thenYield.erase();

  builder.setInsertionPoint(elseYield);
  llvm::SmallVector<mlir::Value> newElseResults;
  for (unsigned i = 0; i < numArgs; ++i) {
    if (i == minMaxIdx)
      continue;
    newElseResults.push_back(elseYield->getOperand(i));
  }
  fir::ResultOp::create(builder, loc, newElseResults);
  elseYield.erase();

  // Replace fir.if condition with (mask AND isFirst).
  mlir::Value origMaskCond = ifOp.getCondition();
  if (origMaskCond.getType() != builder.getI1Type()) {
    while (auto conv = origMaskCond.getDefiningOp<fir::ConvertOp>())
      origMaskCond = conv.getOperand();
    if (origMaskCond.getType() != builder.getI1Type()) {
      builder.setInsertionPoint(ifOp);
      origMaskCond = fir::ConvertOp::create(builder, loc, builder.getI1Type(),
                                            origMaskCond);
    }
  }
  builder.setInsertionPoint(ifOp);
  mlir::Value isFirstIterArg = innerLoop.getRegionIterArgs()[isFirstIdx];
  mlir::Value newIfCond =
      mlir::arith::AndIOp::create(builder, loc, origMaskCond, isFirstIterArg);
  // Replace fir.if with one that returns reduced result types (no minMax).
  llvm::SmallVector<mlir::Type> newResultTypes;
  for (unsigned i = 0; i < numArgs; ++i) {
    if (i != minMaxIdx)
      newResultTypes.push_back(ifOp.getResultTypes()[i]);
  }
  auto newIfOp = fir::IfOp::create(builder, loc, newResultTypes, newIfCond,
                                   /*withElseRegion=*/true);
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());

  // Remove dead ops left in the then-block after minMax was dropped.
  // Iterates until fixed point because erasing one op may expose the next.
  mlir::Block &newThenBlock = newIfOp.getThenRegion().front();
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &op : llvm::make_early_inc_range(newThenBlock)) {
      if (!mlir::isa<fir::ResultOp>(op) && mlir::isOpTriviallyDead(&op)) {
        op.erase();
        changed = true;
      }
    }
  }

  // Save terminators before any splicing.
  llvm::SmallVector<mlir::Operation *> nestTerminators;
  for (fir::DoLoopOp loop : nest)
    nestTerminators.push_back(loop.getBody()->getTerminator());

  // Replace innermost terminator with newIfOp results.
  auto innerTerminator = nestTerminators.back();
  builder.setInsertionPoint(innerTerminator);
  mlir::Operation *newInnerTerminator =
      fir::ResultOp::create(builder, loc, newIfOp.getResults()).getOperation();
  innerTerminator->erase();
  ifOp.erase();

  // Rebuild each loop in the nest without the minMax iter arg.
  fir::DoLoopOp prevTransformedLoop = nullptr;
  for (int i = static_cast<int>(nest.size()) - 1; i >= 0; --i) {
    fir::DoLoopOp loop = nest[i];
    mlir::Block *oldBody = loop.getBody();
    mlir::Operation *oldTerm = (i == static_cast<int>(nest.size()) - 1)
                                   ? newInnerTerminator
                                   : nestTerminators[i];

    llvm::SmallVector<mlir::Value> newLoopInitArgs;
    for (unsigned j = 0; j < numArgs; ++j) {
      if (j != minMaxIdx)
        newLoopInitArgs.push_back(loop.getInitArgs()[j]);
    }

    builder.setInsertionPoint(loop);
    auto transformedLoop = fir::DoLoopOp::create(
        builder, loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
        loop.getStep(), loop.getUnordered().value_or(false),
        /*finalValue=*/false, newLoopInitArgs);

    mlir::Block *newBody = transformedLoop.getBody();

    newBody->getOperations().splice(newBody->begin(), oldBody->getOperations(),
                                    oldBody->begin(), oldTerm->getIterator());

    oldBody->getArgument(0).replaceAllUsesWith(newBody->getArgument(0));
    unsigned newArgIdx = 1;
    for (unsigned j = 0; j < numArgs; ++j) {
      if (j != minMaxIdx)
        oldBody->getArgument(j + 1).replaceAllUsesWith(
            newBody->getArgument(newArgIdx++));
    }

    // Terminate: innermost uses newIfOp results, outer uses inner loop results.
    builder.setInsertionPointToEnd(newBody);
    mlir::ValueRange termResults = prevTransformedLoop
                                       ? prevTransformedLoop.getResults()
                                       : newIfOp.getResults();
    fir::ResultOp::create(builder, loc, termResults);
    oldTerm->erase();

    unsigned trackingIdx = 0;
    for (unsigned j = 0; j < numArgs; ++j) {
      if (j == minMaxIdx) {
        assert((loop != rootLoop || loop.getResult(j).use_empty()) &&
               "minMax result must be unused outside the nest");
        loop.getResult(j).replaceAllUsesWith(loop.getInitArgs()[minMaxIdx]);
      } else
        loop.getResult(j).replaceAllUsesWith(
            transformedLoop.getResult(trackingIdx++));
    }

    prevTransformedLoop = transformedLoop;
    loop.erase();
  }

  return true;
}

class LoopIdiomRecognizePass
    : public fir::impl::LoopIdiomRecognizeBase<LoopIdiomRecognizePass> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    fir::FirOpBuilder builder(func, fir::getKindMapping(func));

    llvm::SmallVector<fir::DoLoopOp> candidateLoops;
    func.walk<mlir::WalkOrder::PostOrder>(
        [&](fir::DoLoopOp loop) { candidateLoops.push_back(loop); });

    for (fir::DoLoopOp loop : candidateLoops)
      transformEqualityMaskMinMaxLoop(loop, builder);
  }
};

} // namespace
