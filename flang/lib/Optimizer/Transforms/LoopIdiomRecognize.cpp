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
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace fir {
#define GEN_PASS_DEF_LOOPIDIOMRECOGNIZE
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

/// If cond is arith.cmpi eq with elemVal on one side and a loop-invariant
/// value on the other, returns the invariant operand; otherwise nullptr.
static mlir::Value getFIREqualityMaskTarget(mlir::Value cond,
                                            mlir::Value elemVal,
                                            fir::DoLoopOp rootLoop) {
  auto isInvariant = [&](mlir::Value v) {
    if (mlir::Operation *def = v.getDefiningOp())
      return !rootLoop->isAncestor(def);
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      mlir::Operation *owner = arg.getOwner()->getParentOp();
      return !rootLoop->isAncestor(owner) && (owner != rootLoop.getOperation());
    }
    return true;
  };

  while (auto conv = cond.getDefiningOp<fir::ConvertOp>())
    cond = conv.getOperand();

  auto cmpOp = cond.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmpOp || cmpOp.getPredicate() != mlir::arith::CmpIPredicate::eq)
    return nullptr;

  if (cmpOp.getLhs() == elemVal && isInvariant(cmpOp.getRhs()))
    return cmpOp.getRhs();
  if (cmpOp.getRhs() == elemVal && isInvariant(cmpOp.getLhs()))
    return cmpOp.getLhs();
  return nullptr;
}

/// Collect pure ops in thenRegion (not nested) needed to compute val, in
/// def-before-use order. Returns false if any required op is impure or nested.
static bool collectLocOps(mlir::Value val, mlir::Region *thenRegion,
                          fir::DoLoopOp innerLoop,
                          llvm::SmallVectorImpl<mlir::Operation *> &ops) {
  mlir::Operation *def = val.getDefiningOp();
  if (!def) {
    if (llvm::is_contained(innerLoop.getRegionIterArgs(), val))
      return false;
    return true;
  }
  if (def->getParentRegion() != thenRegion) {
    if (thenRegion->isAncestor(def->getParentRegion()))
      return false;
    return true;
  }
  if (llvm::is_contained(ops, def))
    return true;
  if (!mlir::isPure(def))
    return false;
  for (mlir::Value operand : def->getOperands())
    if (!collectLocOps(operand, thenRegion, innerLoop, ops))
      return false;
  ops.push_back(def);
  return true;
}

/// Drop the redundant minMax iter arg from an equality-mask location-search
/// loop nest.  The original fir.if is cloned intact to preserve side effects;
/// a new fir.if guarded by (mask AND isFirst) tracks only location/isFirst.
/// The cloned fir.if is erased if side-effect-free, otherwise left for DCE.
static bool transformEqualityMaskMinMaxLoop(
    fir::DoLoopOp rootLoop, fir::FirOpBuilder &builder,
    llvm::DenseSet<mlir::Operation *> &erasedLoops) {
  unsigned numArgs = rootLoop.getInitArgs().size();
  if (numArgs < 3)
    return false;
  if (rootLoop.getFinalValue().value_or(false))
    return false;

  unsigned isFirstIdx = numArgs - 1;
  unsigned minMaxIdx = numArgs - 2;

  if (!rootLoop.getResult(minMaxIdx).use_empty())
    return false;

  if (!rootLoop.getInitArgs()[isFirstIdx].getType().isInteger(1))
    return false;
  auto isFirstConst = rootLoop.getInitArgs()[isFirstIdx]
                          .getDefiningOp<mlir::arith::ConstantOp>();
  if (!isFirstConst)
    return false;
  auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(isFirstConst.getValue());
  if (!boolAttr || !boolAttr.getValue())
    return false;

  // Only transform from the outermost loop of a nest.
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

  for (unsigned i = 0; i + 1 < nest.size(); ++i) {
    mlir::Value outerArg = nest[i].getRegionIterArgs()[minMaxIdx];
    for (mlir::Operation *user : outerArg.getUsers()) {
      if (user != nest[i + 1].getOperation() &&
          user != nest[i].getBody()->getTerminator())
        return false;
    }
    if (nest[i + 1].getInitArgs()[minMaxIdx] != outerArg)
      return false;
    mlir::Value innerRes = nest[i + 1].getResult(minMaxIdx);
    for (mlir::Operation *user : innerRes.getUsers()) {
      if (user != nest[i].getBody()->getTerminator())
        return false;
    }
    // Ensure the outer loop terminator strictly forwards the inner loop
    // results.
    auto outerYield =
        mlir::dyn_cast<fir::ResultOp>(nest[i].getBody()->getTerminator());
    if (!outerYield || outerYield.getNumOperands() != numArgs)
      return false;
    for (unsigned j = 0; j < numArgs; ++j)
      if (outerYield.getOperand(j) != nest[i + 1].getResult(j))
        return false;
  }

  fir::DoLoopOp innerLoop = nest.back();

  auto innerYield =
      mlir::dyn_cast<fir::ResultOp>(innerLoop.getBody()->getTerminator());
  if (!innerYield || innerYield.getNumOperands() != numArgs)
    return false;

  mlir::Value isFirstYieldVal = innerYield.getOperand(isFirstIdx);
  auto ifOp = isFirstYieldVal.getDefiningOp<fir::IfOp>();
  if (!ifOp || !ifOp.getThenRegion().hasOneBlock() ||
      !ifOp.getElseRegion().hasOneBlock())
    return false;
  if (ifOp->getParentRegion() != innerLoop.getBody()->getParent())
    return false;
  if (ifOp.getNumResults() != numArgs)
    return false;
  for (unsigned i = 0; i < numArgs; ++i)
    if (innerYield.getOperand(i) != ifOp.getResult(i))
      return false;

  mlir::Block *thenBlock = &ifOp.getThenRegion().front();
  auto thenYield = mlir::dyn_cast<fir::ResultOp>(thenBlock->getTerminator());
  if (!thenYield || thenYield.getNumOperands() != numArgs)
    return false;

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

  mlir::ValueRange elseArgs = innerLoop.getRegionIterArgs();
  for (unsigned i = 0; i < numArgs; ++i)
    if (elseYield.getOperand(i) != elseArgs[i])
      return false;

  mlir::Value minMaxVal = thenYield.getOperand(minMaxIdx);
  mlir::Value minMaxArg = innerLoop.getRegionIterArgs()[minMaxIdx];
  while (auto sel = minMaxVal.getDefiningOp<mlir::arith::SelectOp>()) {
    if (sel.getFalseValue() != minMaxArg)
      return false;
    minMaxVal = sel.getTrueValue();
  }
  if (llvm::is_contained(innerLoop.getRegionIterArgs(), minMaxVal))
    return false;

  if (auto load = minMaxVal.getDefiningOp<fir::LoadOp>())
    if (fir::isa_volatile_type(load.getMemref().getType()) ||
        fir::isa_volatile_type(load.getType()))
      return false;

  if (!getFIREqualityMaskTarget(ifOp.getCondition(), minMaxVal, rootLoop))
    return false;

  for (mlir::Operation *user : minMaxArg.getUsers()) {
    if (!ifOp->isAncestor(user)) {
      if (user != innerYield.getOperation())
        return false;
      continue;
    }
    bool isAllowed = false;
    if (mlir::isa<mlir::arith::CmpIOp>(user) &&
        ifOp.getThenRegion().isAncestor(user->getParentRegion()))
      isAllowed = true;
    else if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(user))
      isAllowed = (sel.getFalseValue() == minMaxArg &&
                   ifOp.getThenRegion().isAncestor(user->getParentRegion()));
    else if (auto res = mlir::dyn_cast<fir::ResultOp>(user))
      isAllowed = (res.getOperation() == elseYield.getOperation());
    if (!isAllowed)
      return false;
  }

  llvm::SmallVector<mlir::Value> trueLocValues;
  llvm::SmallVector<mlir::Operation *> unifiedLocOps;
  for (unsigned i = 0; i < numArgs; ++i) {
    if (i == minMaxIdx || i == isFirstIdx)
      continue;
    mlir::Value val = thenYield.getOperand(i);
    mlir::Value locIterArg = innerLoop.getRegionIterArgs()[i];
    while (auto sel = val.getDefiningOp<mlir::arith::SelectOp>()) {
      if (sel.getFalseValue() != locIterArg)
        return false;
      val = sel.getTrueValue();
    }
    if (llvm::is_contained(innerLoop.getRegionIterArgs(), val))
      return false;
    if (!collectLocOps(val, &ifOp.getThenRegion(), innerLoop, unifiedLocOps))
      return false;
    trueLocValues.push_back(val);
  }

  mlir::Location loc = rootLoop.getLoc();

  mlir::Value origMaskCond = ifOp.getCondition();
  while (auto conv = origMaskCond.getDefiningOp<fir::ConvertOp>())
    origMaskCond = conv.getOperand();

  llvm::SmallVector<llvm::SmallVector<mlir::Value>> nestInitArgs;
  for (fir::DoLoopOp loop : nest)
    nestInitArgs.push_back(llvm::SmallVector<mlir::Value>(loop.getInitArgs()));

  llvm::SmallVector<fir::DoLoopOp> transformedNest(nest.size());
  fir::IfOp newIfOp;
  mlir::IRMapping globalMap;

  // The dropped minMax arg is mapped to its init constant so cloned ops stay
  // valid.
  auto createLoopShell = [&](unsigned nestIdx) {
    fir::DoLoopOp loop = nest[nestIdx];
    mlir::Block *oldBody = loop.getBody();

    llvm::SmallVector<mlir::Value> newLoopInitArgs;
    for (unsigned j = 0; j < numArgs; ++j) {
      if (j == minMaxIdx)
        continue;
      newLoopInitArgs.push_back(
          globalMap.lookupOrDefault(nestInitArgs[nestIdx][j]));
    }

    auto transformedLoop = fir::DoLoopOp::create(
        builder, loop.getLoc(), globalMap.lookupOrDefault(loop.getLowerBound()),
        globalMap.lookupOrDefault(loop.getUpperBound()),
        globalMap.lookupOrDefault(loop.getStep()),
        loop.getUnordered().value_or(false),
        /*finalValue=*/false, newLoopInitArgs);
    transformedNest[nestIdx] = transformedLoop;

    mlir::Block *newBody = transformedLoop.getBody();
    globalMap.map(oldBody->getArgument(0), newBody->getArgument(0));
    unsigned newArgIdx = 1;
    for (unsigned j = 0; j < numArgs; ++j) {
      if (j == minMaxIdx)
        globalMap.map(
            oldBody->getArgument(j + 1),
            globalMap.lookupOrDefault(nestInitArgs[nestIdx][minMaxIdx]));
      else
        globalMap.map(oldBody->getArgument(j + 1),
                      newBody->getArgument(newArgIdx++));
    }
    builder.setInsertionPointToEnd(newBody);
  };

  builder.setInsertionPoint(nest[0]);
  createLoopShell(0);

  for (unsigned i = 0; i + 1 < nest.size(); ++i) {
    fir::DoLoopOp oldInnerLoop = nest[i + 1];
    mlir::Block *oldBody = nest[i].getBody();
    mlir::Operation *oldTerm = oldBody->getTerminator();

    builder.setInsertionPointToEnd(transformedNest[i].getBody());

    for (auto it = oldBody->begin();
         &*it != oldTerm && &*it != oldInnerLoop.getOperation(); ++it)
      builder.clone(*it, globalMap);

    createLoopShell(i + 1);

    unsigned trackingIdx = 0;
    for (unsigned j = 0; j < numArgs; ++j) {
      if (j == minMaxIdx)
        globalMap.map(
            oldInnerLoop.getResult(j),
            globalMap.lookupOrDefault(nestInitArgs[i + 1][minMaxIdx]));
      else
        globalMap.map(oldInnerLoop.getResult(j),
                      transformedNest[i + 1].getResult(trackingIdx++));
    }

    builder.setInsertionPointToEnd(transformedNest[i].getBody());
    for (auto it = std::next(oldInnerLoop->getIterator()); &*it != oldTerm;
         ++it)
      builder.clone(*it, globalMap);
  }

  unsigned inner = nest.size() - 1;
  mlir::Block *oldBody = nest[inner].getBody();
  mlir::Operation *oldTerm = oldBody->getTerminator();
  builder.setInsertionPointToEnd(transformedNest[inner].getBody());

  for (auto it = oldBody->begin(); &*it != oldTerm; ++it)
    builder.clone(*it, globalMap);

  mlir::Operation *clonedIfBase = globalMap.lookup(ifOp.getOperation());
  assert(clonedIfBase && "cloned ifOp not found in globalMap");

  mlir::ValueRange newIterArgs = transformedNest[inner].getRegionIterArgs();
  mlir::Value isFirstArg = newIterArgs[newIterArgs.size() - 1];

  builder.setInsertionPointAfter(clonedIfBase);

  mlir::Value mappedMaskCond = globalMap.lookup(origMaskCond);
  assert(mappedMaskCond && "mask condition not found in cloned loop body");
  if (mappedMaskCond.getType() != builder.getI1Type())
    mappedMaskCond = fir::ConvertOp::create(builder, loc, builder.getI1Type(),
                                            mappedMaskCond);
  mlir::Value newIfCond =
      mlir::arith::AndIOp::create(builder, loc, mappedMaskCond, isFirstArg);

  llvm::SmallVector<mlir::Type> newResultTypes;
  for (unsigned j = 0; j < numArgs; ++j)
    if (j != minMaxIdx)
      newResultTypes.push_back(ifOp.getResultTypes()[j]);

  newIfOp = fir::IfOp::create(builder, loc, newResultTypes, newIfCond,
                              /*withElseRegion=*/true);

  mlir::Block &thenBlk = newIfOp.getThenRegion().front();
  mlir::Block &elseBlk = newIfOp.getElseRegion().front();

  builder.setInsertionPointToEnd(&thenBlk);

  // Build locMap from globalMap, skipping then-region ops cloned separately.
  mlir::IRMapping locMap;
  for (auto [oldVal, newVal] : globalMap.getValueMap()) {
    if (auto oldOp = oldVal.getDefiningOp())
      if (ifOp.getThenRegion().isAncestor(oldOp->getParentRegion()))
        continue;
    locMap.map(oldVal, newVal);
  }

  for (mlir::Operation *op : unifiedLocOps) {
    if (op->getNumResults() > 0 && locMap.contains(op->getResult(0)))
      continue;
    builder.clone(*op, locMap);
  }

  llvm::SmallVector<mlir::Value> newLocValues;
  for (mlir::Value val : trueLocValues) {
    mlir::Operation *def = val.getDefiningOp();
    if (def && def->getParentRegion() == &ifOp.getThenRegion())
      newLocValues.push_back(locMap.lookup(val));
    else
      newLocValues.push_back(globalMap.lookupOrDefault(val));
  }

  mlir::Value falseVal = builder.createBool(loc, false);

  llvm::SmallVector<mlir::Value> newThenVals;
  unsigned locIdx = 0;
  for (unsigned j = 0; j < numArgs; ++j) {
    if (j == minMaxIdx)
      continue;
    newThenVals.push_back(j == isFirstIdx ? falseVal : newLocValues[locIdx++]);
  }
  fir::ResultOp::create(builder, loc, newThenVals);

  builder.setInsertionPointToEnd(&elseBlk);
  fir::ResultOp::create(builder, loc,
                        llvm::SmallVector<mlir::Value>(newIterArgs));

  // Erase the cloned fir.if if pure - its results are unused. If it has side
  // effects, leave it so they still fire on mask-true iterations.
  auto clonedIf = mlir::cast<fir::IfOp>(clonedIfBase);
  if (mlir::isMemoryEffectFree(clonedIf.getOperation()))
    clonedIf.erase();
  for (int i = static_cast<int>(nest.size()) - 1; i >= 0; --i) {
    builder.setInsertionPointToEnd(transformedNest[i].getBody());
    mlir::ValueRange termResults = (i == static_cast<int>(nest.size()) - 1)
                                       ? newIfOp.getResults()
                                       : transformedNest[i + 1].getResults();
    fir::ResultOp::create(builder, loc, termResults);
  }

  // Erase outermost-first - erasing innermost first leaves dangling uses.
  fir::DoLoopOp outerTransformed = transformedNest[0];
  unsigned trackingIdx = 0;
  for (unsigned j = 0; j < numArgs; ++j) {
    if (j == minMaxIdx)
      nest[0].getResult(j).replaceAllUsesWith(nestInitArgs[0][minMaxIdx]);
    else
      nest[0].getResult(j).replaceAllUsesWith(
          outerTransformed.getResult(trackingIdx++));
  }
  for (fir::DoLoopOp loopInNest : nest)
    erasedLoops.insert(loopInNest.getOperation());
  nest[0].erase();

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

    llvm::DenseSet<mlir::Operation *> erasedLoops;
    for (fir::DoLoopOp loop : candidateLoops) {
      if (erasedLoops.contains(loop.getOperation()))
        continue;
      transformEqualityMaskMinMaxLoop(loop, builder, erasedLoops);
    }
  }
};

} // namespace
