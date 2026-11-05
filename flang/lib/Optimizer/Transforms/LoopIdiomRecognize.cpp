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
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(v))
      return !rootLoop->isAncestor(arg.getOwner()->getParentOp());
    llvm_unreachable("expected op result or block argument");
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

/// Collect pure ops in thenRegion needed to compute val, in def-before-use
/// order. Returns false if any required op is impure or loop-carried.
static bool collectLocOps(mlir::Value val, mlir::Region *thenRegion,
                          fir::DoLoopOp innerLoop,
                          llvm::SmallVectorImpl<mlir::Operation *> &ops,
                          llvm::DenseSet<mlir::Operation *> &visited) {
  mlir::Operation *def = val.getDefiningOp();
  if (!def) {
    if (llvm::is_contained(innerLoop.getRegionIterArgs(), val))
      return false;
    return true;
  }
  if (def->getParentRegion() != thenRegion)
    return true;
  // Ops shared between loc values are collected, and later cloned, only once.
  if (!visited.insert(def).second)
    return true;
  if (!mlir::isPure(def))
    return false;
  for (mlir::Value operand : def->getOperands())
    if (!collectLocOps(operand, thenRegion, innerLoop, ops, visited))
      return false;
  ops.push_back(def);
  return true;
}

/// Match the fir.if inside innerLoop performing equality-mask min/max search:
///   fir.if (x == invariant) {
///     isFirst = false; loc = select(upd, ..); minMax = select(upd, x, minMax)
///   } else { passthrough all iter args }
/// where x is the element being reduced, minMax has no other uses, and the
/// fir.if results feed only the loop yield.
static fir::IfOp matchEqualityMaskSearchIf(
    fir::DoLoopOp innerLoop, fir::DoLoopOp rootLoop, unsigned minMaxIdx,
    unsigned isFirstIdx, fir::ResultOp &thenYieldOut, mlir::Value &updCondOut) {
  auto innerYield =
      mlir::dyn_cast<fir::ResultOp>(innerLoop.getBody()->getTerminator());
  if (!innerYield)
    return nullptr;

  // The innermost yield must come straight from a single fir.if.
  mlir::Value isFirstYieldVal = innerYield.getOperand(isFirstIdx);
  auto ifOp = isFirstYieldVal.getDefiningOp<fir::IfOp>();
  if (!ifOp || !ifOp.getThenRegion().hasOneBlock() ||
      !ifOp.getElseRegion().hasOneBlock())
    return nullptr;
  if (!llvm::equal(innerYield.getOperands(), ifOp.getResults()))
    return nullptr;

  mlir::Block *thenBlock = &ifOp.getThenRegion().front();
  auto thenYield = mlir::dyn_cast<fir::ResultOp>(thenBlock->getTerminator());
  if (!thenYield)
    return nullptr;

  // The then-branch clears isFirst.
  auto thenIsFirstConst =
      thenYield.getOperand(isFirstIdx).getDefiningOp<mlir::arith::ConstantOp>();
  if (!thenIsFirstConst)
    return nullptr;
  auto thenBoolAttr =
      mlir::dyn_cast<mlir::BoolAttr>(thenIsFirstConst.getValue());
  if (!thenBoolAttr || thenBoolAttr.getValue())
    return nullptr;

  // The else-branch forwards all iter args unchanged.
  mlir::Block *elseBlock = &ifOp.getElseRegion().front();
  auto elseYield = mlir::dyn_cast<fir::ResultOp>(elseBlock->getTerminator());
  if (!elseYield)
    return nullptr;

  if (!llvm::equal(elseYield.getOperands(), innerLoop.getRegionIterArgs()))
    return nullptr;

  // minMax = select(upd, x, minMax), where x is the element being reduced.
  mlir::Value minMaxArg = innerLoop.getRegionIterArgs()[minMaxIdx];
  auto minMaxSel =
      thenYield.getOperand(minMaxIdx).getDefiningOp<mlir::arith::SelectOp>();
  if (!minMaxSel || minMaxSel.getFalseValue() != minMaxArg)
    return nullptr;
  mlir::Value minMaxVal = minMaxSel.getTrueValue();
  // x must not be loop-carried.
  if (llvm::is_contained(innerLoop.getRegionIterArgs(), minMaxVal))
    return nullptr;

  // Keep volatile accesses as they are.
  if (auto load = minMaxVal.getDefiningOp<fir::LoadOp>())
    if (fir::isa_volatile_type(load.getMemref().getType()) ||
        fir::isa_volatile_type(load.getType()))
      return nullptr;

  // The mask is x == invariant, so after the first match minMax equals the
  // invariant and a strict compare can't succeed again.
  if (!getFIREqualityMaskTarget(ifOp.getCondition(), minMaxVal, rootLoop))
    return nullptr;

  // upd = (minMaxVal <strict> minMaxArg) | isFirst
  auto updOr = minMaxSel.getCondition().getDefiningOp<mlir::arith::OrIOp>();
  if (!updOr)
    return nullptr;
  mlir::Value isFirstArg = innerLoop.getRegionIterArgs()[isFirstIdx];
  mlir::Value cmpVal;
  if (updOr.getRhs() == isFirstArg)
    cmpVal = updOr.getLhs();
  else if (updOr.getLhs() == isFirstArg)
    cmpVal = updOr.getRhs();
  else
    return nullptr;
  auto cmp = cmpVal.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmp)
    return nullptr;
  using Pred = mlir::arith::CmpIPredicate;
  Pred pred = cmp.getPredicate();
  if (pred != Pred::slt && pred != Pred::sgt && pred != Pred::ult &&
      pred != Pred::ugt)
    return nullptr;
  if (!((cmp.getLhs() == minMaxVal && cmp.getRhs() == minMaxArg) ||
        (cmp.getLhs() == minMaxArg && cmp.getRhs() == minMaxVal)))
    return nullptr;

  // The if results may only feed the loop yield.
  for (mlir::Operation *user : ifOp->getUsers())
    if (user != innerYield.getOperation())
      return nullptr;

  // minMaxArg may only feed the compare/select in the then-region and the
  // else passthrough.
  for (mlir::Operation *user : minMaxArg.getUsers()) {
    if (!ifOp->isAncestor(user))
      return nullptr;
    bool isAllowed = false;
    if ((user == cmp.getOperation() || user == minMaxSel.getOperation()) &&
        ifOp.getThenRegion().isAncestor(user->getParentRegion()))
      isAllowed = true;
    else if (auto res = mlir::dyn_cast<fir::ResultOp>(user))
      isAllowed = (res.getOperation() == elseYield.getOperation());
    if (!isAllowed)
      return nullptr;
  }

  thenYieldOut = thenYield;
  updCondOut = minMaxSel.getCondition();
  return ifOp;
}

/// Per-level state collected while following an iter-arg chain through a nest.
struct NestLevel {
  fir::DoLoopOp loop;
  unsigned iterIdx; // iter arg position of the tracked chain at this level
};

/// A root iter arg followed down to a fir.if result. droppable is false if an
/// intermediate value has uses besides forwarding through the nest.
struct Chain {
  llvm::SmallVector<NestLevel> levels;
  bool droppable;
};

struct Candidate {
  llvm::SmallVector<NestLevel> levels;
  bool isFirst;
};

struct LocIdiom {
  llvm::SmallVector<NestLevel> nest;
  fir::IfOp ifOp;
  unsigned isFirstIdx; // isFirst position in the innermost loop
  llvm::SmallVector<mlir::Value> trueLocValues;
  llvm::SmallVector<mlir::Operation *> unifiedLocOps;
};

/// Find an isFirst and a minMax candidate that share a loop nest and meet in
/// one fir.if of the equality-mask idiom. Returns nullopt if no fully valid
/// pair exists (structural match, effect-free, loc selects all consistent).
static std::optional<LocIdiom>
matchLocIdiom(llvm::ArrayRef<Candidate> candidates, fir::DoLoopOp rootLoop) {
  auto loopOf = [](const NestLevel &l) { return l.loop; };
  for (const Candidate &isFirstCand : candidates) {
    if (!isFirstCand.isFirst)
      continue;
    for (const Candidate &minMaxCand : candidates) {
      if (minMaxCand.isFirst)
        continue;
      if (!llvm::equal(llvm::map_range(minMaxCand.levels, loopOf),
                       llvm::map_range(isFirstCand.levels, loopOf)))
        continue;
      fir::DoLoopOp innerLoop = minMaxCand.levels.back().loop;
      unsigned innerNumArgs = innerLoop.getInitArgs().size();
      unsigned minMaxIdx = minMaxCand.levels.back().iterIdx;
      unsigned isFirstIdx = isFirstCand.levels.back().iterIdx;
      fir::ResultOp thenYield;
      mlir::Value updCond;
      fir::IfOp ifOp = matchEqualityMaskSearchIf(
          innerLoop, rootLoop, minMaxIdx, isFirstIdx, thenYield, updCond);
      if (!ifOp)
        continue;
      if (!mlir::isMemoryEffectFree(ifOp.getOperation()))
        continue;
      llvm::SmallVector<mlir::Value> trueLocValues;
      llvm::SmallVector<mlir::Operation *> unifiedLocOps;
      llvm::DenseSet<mlir::Operation *> locVisited;
      bool locOk = true;
      for (unsigned i = 0; i < innerNumArgs; ++i) {
        if (i == minMaxIdx || i == isFirstIdx)
          continue;
        auto sel =
            thenYield.getOperand(i).getDefiningOp<mlir::arith::SelectOp>();
        // Each loc must be select(upd, newLoc, oldLoc).
        if (!sel || sel.getCondition() != updCond ||
            sel.getFalseValue() != innerLoop.getRegionIterArgs()[i]) {
          locOk = false;
          break;
        }
        mlir::Value val = sel.getTrueValue();
        if (!collectLocOps(val, &ifOp.getThenRegion(), innerLoop, unifiedLocOps,
                           locVisited)) {
          locOk = false;
          break;
        }
        trueLocValues.push_back(val);
      }
      if (!locOk)
        continue;
      return LocIdiom{minMaxCand.levels, ifOp, isFirstIdx,
                      std::move(trueLocValues), std::move(unifiedLocOps)};
    }
  }
  return std::nullopt;
}

/// Drop the redundant minMax iter arg from an equality-mask location-search
/// loop nest. Only the first match can change loc, so the original fir.if is
/// replaced by one guarded by (mask AND isFirst). The cloned fir.if only
/// anchors the insertion point and is erased.
static bool transformEqualityMaskMinMaxLoop(fir::DoLoopOp rootLoop,
                                            fir::FirOpBuilder &builder) {
  // A final value shifts the result numbering and reduce operands are not
  // tracked, so only plain loops are handled.
  auto isPlainLoop = [](fir::DoLoopOp loop) {
    return !loop.getFinalValue().value_or(false) &&
           loop.getNumReduceOperands() == 0;
  };
  if (!isPlainLoop(rootLoop))
    return false;

  unsigned rootNumArgs = rootLoop.getInitArgs().size();

  // Follow the chain for root iter arg rootIdx through nested loops until a
  // fir.if result is reached.
  auto followChain = [&](unsigned rootIdx) -> std::optional<Chain> {
    llvm::SmallVector<NestLevel> levels;
    bool droppable = true;
    fir::DoLoopOp curr = rootLoop;
    unsigned idx = rootIdx;
    while (true) {
      auto term =
          mlir::dyn_cast<fir::ResultOp>(curr.getBody()->getTerminator());
      if (!term)
        return std::nullopt;
      mlir::Value operand = term.getOperand(idx);
      if (auto innerLoop =
              mlir::dyn_cast_or_null<fir::DoLoopOp>(operand.getDefiningOp())) {
        auto result = mlir::cast<mlir::OpResult>(operand);
        if (!isPlainLoop(innerLoop))
          return std::nullopt;
        unsigned innerIdx = result.getResultNumber();
        mlir::Value iterArg = curr.getRegionIterArgs()[idx];
        // The iter arg may only seed innerLoop's init slot. Forwarding it
        // elsewhere (e.g. a different yield slot) would break once the arg is
        // remapped.
        if (innerLoop.getInitArgs()[innerIdx] != iterArg)
          return std::nullopt;
        // A dropped result is replaced by its init value, so any use other
        // than this level's yield slot would see the wrong value.
        if (!iterArg.hasOneUse() || !result.hasOneUse())
          droppable = false;
        levels.push_back({curr, idx});
        curr = innerLoop;
        idx = innerIdx;
        continue;
      }
      if (auto ifResult = mlir::dyn_cast<mlir::OpResult>(operand))
        if (mlir::isa<fir::IfOp>(ifResult.getOwner())) {
          levels.push_back({curr, idx});
          return Chain{std::move(levels), droppable};
        }
      return std::nullopt;
    }
  };

  llvm::SmallVector<Candidate> candidates;

  for (unsigned j = 0; j < rootNumArgs; ++j) {
    auto chainResult = followChain(j);
    if (!chainResult)
      continue;
    auto &[levels, droppable] = *chainResult;

    // isFirst candidate: root init is i1 true (then-branch sets it false).
    // minMax candidate: root result is unused outside the loop nest.
    mlir::Value rootInit = rootLoop.getInitArgs()[j];
    if (rootInit.getType().isInteger(1)) {
      if (auto cst = rootInit.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto ba = mlir::dyn_cast<mlir::BoolAttr>(cst.getValue())) {
          if (ba.getValue()) {
            candidates.push_back({std::move(levels), true});
            continue;
          }
        }
      }
    }

    if (droppable && rootLoop.getResult(j).use_empty())
      candidates.push_back({std::move(levels), false});
  }

  auto idiom = matchLocIdiom(candidates, rootLoop);
  if (!idiom)
    return false;
  llvm::SmallVector<NestLevel> &nest = idiom->nest;
  fir::IfOp ifOp = idiom->ifOp;
  llvm::SmallVector<mlir::Value> &trueLocValues = idiom->trueLocValues;
  llvm::SmallVector<mlir::Operation *> &unifiedLocOps = idiom->unifiedLocOps;

  unsigned numArgs = nest.back().loop.getInitArgs().size();
  unsigned minMaxIdx = nest.back().iterIdx;
  unsigned isFirstIdx = idiom->isFirstIdx;

  mlir::Location loc = rootLoop.getLoc();

  llvm::SmallVector<llvm::SmallVector<mlir::Value>> nestInitArgs;
  for (auto &level : nest)
    nestInitArgs.push_back(
        llvm::SmallVector<mlir::Value>(level.loop.getInitArgs()));

  llvm::SmallVector<fir::DoLoopOp> transformedNest(nest.size());
  fir::IfOp newIfOp;
  mlir::IRMapping globalMap;

  auto createLoopShell = [&](unsigned nestIdx) {
    fir::DoLoopOp loop = nest[nestIdx].loop;
    unsigned levelMinMaxIdx = nest[nestIdx].iterIdx;
    unsigned levelNumArgs = loop.getInitArgs().size();

    llvm::SmallVector<mlir::Value> newLoopInitArgs;
    for (unsigned j = 0; j < levelNumArgs; ++j) {
      if (j == levelMinMaxIdx)
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
    transformedLoop->setDiscardableAttrs(loop->getDiscardableAttrDictionary());
    if (auto ann = loop.getLoopAnnotationAttr())
      transformedLoop.setLoopAnnotationAttr(ann);

    globalMap.map(loop.getInductionVar(), transformedLoop.getInductionVar());
    unsigned newArgIdx = 0;
    for (auto [j, oldIterArg] : llvm::enumerate(loop.getRegionIterArgs())) {
      if (j == levelMinMaxIdx)
        globalMap.map(oldIterArg, globalMap.lookupOrDefault(
                                      nestInitArgs[nestIdx][levelMinMaxIdx]));
      else
        globalMap.map(oldIterArg,
                      transformedLoop.getRegionIterArgs()[newArgIdx++]);
    }
    builder.setInsertionPointToEnd(transformedLoop.getBody());
  };

  builder.setInsertionPoint(nest[0].loop);
  createLoopShell(0);

  for (unsigned i = 0; i + 1 < nest.size(); ++i) {
    fir::DoLoopOp oldInnerLoop = nest[i + 1].loop;
    mlir::Block *oldBody = nest[i].loop.getBody();
    mlir::Operation *oldTerm = oldBody->getTerminator();
    unsigned innerMinMaxIdx = nest[i + 1].iterIdx;
    unsigned innerNumArgs = oldInnerLoop.getInitArgs().size();

    builder.setInsertionPointToEnd(transformedNest[i].getBody());

    for (auto it = oldBody->begin();
         &*it != oldTerm && &*it != oldInnerLoop.getOperation(); ++it)
      builder.clone(*it, globalMap);

    createLoopShell(i + 1);

    unsigned trackingIdx = 0;
    for (unsigned j = 0; j < innerNumArgs; ++j) {
      if (j == innerMinMaxIdx)
        globalMap.map(
            oldInnerLoop.getResult(j),
            globalMap.lookupOrDefault(nestInitArgs[i + 1][innerMinMaxIdx]));
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
  mlir::Block *oldBody = nest[inner].loop.getBody();
  mlir::Operation *oldTerm = oldBody->getTerminator();
  builder.setInsertionPointToEnd(transformedNest[inner].getBody());

  for (auto it = oldBody->begin(); &*it != oldTerm; ++it)
    builder.clone(*it, globalMap);

  mlir::Operation *clonedIfBase = globalMap.lookupOrNull(ifOp.getOperation());
  assert(clonedIfBase && "ifOp must have been cloned into innermost body");

  mlir::ValueRange newIterArgs = transformedNest[inner].getRegionIterArgs();
  // isFirst's position once the minMax slot is dropped.
  unsigned newIsFirstIdx = 0;
  for (unsigned j = 0; j < isFirstIdx; ++j)
    if (j != minMaxIdx)
      ++newIsFirstIdx;
  mlir::Value isFirstArg = newIterArgs[newIsFirstIdx];

  builder.setInsertionPointAfter(clonedIfBase);

  // The mask may have been hoisted out of the loop, in which case it is not
  // remapped.
  mlir::Value mappedMaskCond = globalMap.lookupOrDefault(ifOp.getCondition());
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

  // Copy globalMap into locMap, except for values defined inside the original
  // then-region. Those are cloned again into the new fir.if.
  mlir::IRMapping locMap;
  for (auto [oldVal, newVal] : globalMap.getValueMap()) {
    if (auto oldOp = oldVal.getDefiningOp())
      if (ifOp.getThenRegion().isAncestor(oldOp->getParentRegion()))
        continue;
    locMap.map(oldVal, newVal);
  }

  for (mlir::Operation *op : unifiedLocOps)
    builder.clone(*op, locMap);

  llvm::SmallVector<mlir::Value> newLocValues;
  for (mlir::Value val : trueLocValues) {
    mlir::Operation *def = val.getDefiningOp();
    if (def && def->getParentRegion() == &ifOp.getThenRegion()) {
      mlir::Value mapped = locMap.lookupOrNull(val);
      assert(mapped && "loc op defined in then-region must be in locMap");
      newLocValues.push_back(mapped);
    } else {
      newLocValues.push_back(globalMap.lookupOrDefault(val));
    }
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

  mlir::cast<fir::IfOp>(clonedIfBase).erase();
  for (int i = static_cast<int>(nest.size()) - 1; i >= 0; --i) {
    builder.setInsertionPointToEnd(transformedNest[i].getBody());
    if (i == static_cast<int>(nest.size()) - 1) {
      fir::ResultOp::create(builder, loc, newIfOp.getResults());
      continue;
    }
    // Outer levels may carry extra iter args, so rebuild the terminator from
    // the old one rather than forwarding the inner loop results.
    llvm::SmallVector<mlir::Value> termResults;
    for (auto [j, v] : llvm::enumerate(
             nest[i].loop.getBody()->getTerminator()->getOperands()))
      if (j != nest[i].iterIdx)
        termResults.push_back(globalMap.lookupOrDefault(v));
    fir::ResultOp::create(builder, loc, termResults);
  }

  fir::DoLoopOp outerTransformed = transformedNest[0];
  unsigned rootMinMaxIdx = nest[0].iterIdx;
  unsigned trackingIdx = 0;
  // minMax result is unused outside the nest; replace it with its init value.
  for (unsigned j = 0; j < rootNumArgs; ++j) {
    if (j == rootMinMaxIdx)
      nest[0].loop.getResult(j).replaceAllUsesWith(
          nestInitArgs[0][rootMinMaxIdx]);
    else
      nest[0].loop.getResult(j).replaceAllUsesWith(
          outerTransformed.getResult(trackingIdx++));
  }
  nest[0].loop.erase();

  return true;
}

class LoopIdiomRecognizePass
    : public fir::impl::LoopIdiomRecognizeBase<LoopIdiomRecognizePass> {
public:
  void runOnOperation() override {
    auto func = getOperation();
    fir::FirOpBuilder builder(func, fir::getKindMapping(func));

    // Post-order: inner loops are visited before the nests that contain them.
    llvm::SmallVector<fir::DoLoopOp> candidateLoops;
    func.walk<mlir::WalkOrder::PostOrder>(
        [&](fir::DoLoopOp loop) { candidateLoops.push_back(loop); });

    for (fir::DoLoopOp loop : candidateLoops)
      transformEqualityMaskMinMaxLoop(loop, builder);
  }
};

} // namespace
