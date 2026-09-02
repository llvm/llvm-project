//===- LoopAnalysis.cpp - Counted loop recognition -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LoopAnalysis.h"

#include "llvm/ADT/SmallPtrSet.h"

using namespace cir;
using namespace cir::loopopt;

// Bounds compile time on pathological expressions.
static constexpr unsigned kMaxExprDepth = 16;

bool cir::loopopt::isOrdinaryAccess(cir::LoadOp load) {
  return !load.getIsVolatile() && !load.getMemOrder();
}

bool cir::loopopt::isOrdinaryAccess(cir::StoreOp store) {
  return !store.getIsVolatile() && !store.getMemOrder();
}

bool cir::loopopt::isSaturating(mlir::Operation *op) {
  if (auto add = mlir::dyn_cast<cir::AddOp>(op))
    return add.getSaturated();
  if (auto sub = mlir::dyn_cast<cir::SubOp>(op))
    return sub.getSaturated();
  return false;
}

bool cir::loopopt::isOrdinaryLoadOfSlotIn(mlir::Value value, mlir::Value slot,
                                          mlir::Operation *scope) {
  auto load = value.getDefiningOp<cir::LoadOp>();
  return load && isOrdinaryAccess(load) && load.getAddr() == slot &&
         scope->isAncestor(load);
}

static mlir::FailureOr<llvm::APSInt> evalConstant(mlir::Value value,
                                                  unsigned depth);

static bool compatible(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  return lhs.getBitWidth() == rhs.getBitWidth() &&
         lhs.isUnsigned() == rhs.isUnsigned();
}

mlir::FailureOr<llvm::APSInt>
cir::loopopt::checkedAdd(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  if (!compatible(lhs, rhs))
    return mlir::failure();
  bool overflow = false;
  llvm::APInt result = lhs.isUnsigned() ? lhs.uadd_ov(rhs, overflow)
                                        : lhs.sadd_ov(rhs, overflow);
  if (overflow)
    return mlir::failure();
  return llvm::APSInt(result, lhs.isUnsigned());
}

mlir::FailureOr<llvm::APSInt>
cir::loopopt::checkedSub(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  if (!compatible(lhs, rhs))
    return mlir::failure();
  bool overflow = false;
  llvm::APInt result = lhs.isUnsigned() ? lhs.usub_ov(rhs, overflow)
                                        : lhs.ssub_ov(rhs, overflow);
  if (overflow)
    return mlir::failure();
  return llvm::APSInt(result, lhs.isUnsigned());
}

mlir::FailureOr<llvm::APSInt>
cir::loopopt::checkedMul(const llvm::APSInt &lhs, const llvm::APSInt &rhs) {
  if (!compatible(lhs, rhs))
    return mlir::failure();
  bool overflow = false;
  llvm::APInt result = lhs.isUnsigned() ? lhs.umul_ov(rhs, overflow)
                                        : lhs.smul_ov(rhs, overflow);
  if (overflow)
    return mlir::failure();
  return llvm::APSInt(result, lhs.isUnsigned());
}

// Apply an integer operation with the type's own width and signedness,
// failing on any result the type cannot represent.
static mlir::FailureOr<llvm::APSInt> applyChecked(mlir::Operation *op,
                                                  const llvm::APSInt &lhs,
                                                  const llvm::APSInt &rhs) {
  if (mlir::isa<cir::AddOp>(op))
    return checkedAdd(lhs, rhs);
  if (mlir::isa<cir::SubOp>(op))
    return checkedSub(lhs, rhs);
  if (mlir::isa<cir::MulOp>(op))
    return checkedMul(lhs, rhs);
  if (mlir::isa<cir::DivOp>(op)) {
    if (!compatible(lhs, rhs) || rhs.isZero())
      return mlir::failure();
    bool overflow = false;
    llvm::APInt result =
        lhs.isUnsigned() ? lhs.udiv(rhs) : lhs.sdiv_ov(rhs, overflow);
    if (overflow)
      return mlir::failure();
    return llvm::APSInt(result, lhs.isUnsigned());
  }
  return mlir::failure();
}

static mlir::FailureOr<llvm::APSInt> evalConstant(mlir::Value value,
                                                  unsigned depth) {
  if (depth > kMaxExprDepth)
    return mlir::failure();
  auto intType = mlir::dyn_cast<cir::IntType>(value.getType());
  if (!intType)
    return mlir::failure();
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return mlir::failure();

  if (auto constOp = mlir::dyn_cast<cir::ConstantOp>(def)) {
    auto intAttr = mlir::dyn_cast<cir::IntAttr>(constOp.getValue());
    if (!intAttr)
      return mlir::failure();
    return llvm::APSInt(intAttr.getValue(), intType.isUnsigned());
  }

  if (!mlir::isa<cir::AddOp, cir::SubOp, cir::MulOp, cir::DivOp>(def) ||
      isSaturating(def))
    return mlir::failure();
  mlir::FailureOr<llvm::APSInt> lhs =
      evalConstant(def->getOperand(0), depth + 1);
  mlir::FailureOr<llvm::APSInt> rhs =
      evalConstant(def->getOperand(1), depth + 1);
  if (mlir::failed(lhs) || mlir::failed(rhs))
    return mlir::failure();
  return applyChecked(def, *lhs, *rhs);
}

mlir::FailureOr<llvm::APSInt>
cir::loopopt::evaluateConstantIntExpr(mlir::Value value) {
  return evalConstant(value, /*depth=*/0);
}

bool cir::loopopt::isSupportedControlOp(mlir::Operation *op) {
  if (auto load = mlir::dyn_cast<cir::LoadOp>(op))
    return isOrdinaryAccess(load) &&
           mlir::isa<cir::IntType>(load.getResult().getType());
  if (auto constOp = mlir::dyn_cast<cir::ConstantOp>(op))
    return mlir::isa<cir::IntAttr>(constOp.getValue());
  if (mlir::isa<cir::AddOp, cir::SubOp, cir::MulOp>(op))
    return !isSaturating(op) &&
           mlir::isa<cir::IntType>(op->getResult(0).getType());
  // A divide is in the vocabulary only when it folds to a constant.
  if (mlir::isa<cir::DivOp>(op))
    return mlir::succeeded(evaluateConstantIntExpr(op->getResult(0)));
  return false;
}

// Collect the backward slice within block. Values defined outside block
// remain opaque for pattern-specific analysis.
static void collectSliceInBlock(mlir::Value value, mlir::Block &block,
                                llvm::SmallPtrSetImpl<mlir::Operation *> &out) {
  llvm::SmallVector<mlir::Value, 8> worklist{value};
  while (!worklist.empty()) {
    mlir::Operation *def = worklist.pop_back_val().getDefiningOp();
    if (!def || def->getBlock() != &block || !out.insert(def).second)
      continue;
    for (mlir::Value operand : def->getOperands())
      worklist.push_back(operand);
  }
}

static bool
sliceReadsSlot(const llvm::SmallPtrSetImpl<mlir::Operation *> &slice,
               mlir::Value slot) {
  for (mlir::Operation *op : slice)
    if (auto load = mlir::dyn_cast<cir::LoadOp>(op))
      if (load.getAddr() == slot)
        return true;
  return false;
}

// Every write to the slot must be the matched step store or sit outside the
// loop. The alloca use list is complete, so this pass also proves the
// address never escapes.
static bool slotWritesAreOnlyTheStep(mlir::Value ivSlot, cir::StoreOp stepStore,
                                     cir::ForOp forOp) {
  for (mlir::Operation *user : ivSlot.getUsers()) {
    if (mlir::isa<cir::LoadOp>(user))
      continue;
    auto store = mlir::dyn_cast<cir::StoreOp>(user);
    if (!store || store.getAddr() != ivSlot)
      return false;
    if (store != stepStore && forOp->isAncestor(store))
      return false;
  }
  return true;
}

// Require one ordinary initialization store in the loop's parent block
// before the loop. General reaching definitions need dominance analysis.
static cir::StoreOp getInitStore(mlir::Value ivSlot, cir::ForOp forOp) {
  cir::StoreOp init;
  for (mlir::Operation *user : ivSlot.getUsers()) {
    auto store = mlir::dyn_cast<cir::StoreOp>(user);
    if (!store || forOp->isAncestor(store))
      continue;
    if (store->getBlock() != forOp->getBlock() ||
        !store->isBeforeInBlock(forOp) || !isOrdinaryAccess(store) || init)
      return {};
    init = store;
  }
  return init;
}

// The recorded predicate reads as if the induction variable were the left
// operand, so swapping compare operands swaps the recorded kind.
static cir::CmpOpKind normalizeToIvOnLhs(cir::CmpOpKind kind) {
  switch (kind) {
  case cir::CmpOpKind::lt:
    return cir::CmpOpKind::gt;
  case cir::CmpOpKind::gt:
    return cir::CmpOpKind::lt;
  case cir::CmpOpKind::le:
    return cir::CmpOpKind::ge;
  case cir::CmpOpKind::ge:
    return cir::CmpOpKind::le;
  default:
    return kind;
  }
}

// Split the exit test around the induction variable and prove the condition
// region computes nothing else.
static mlir::FailureOr<ControlComparison>
matchControlComparison(cir::ForOp forOp, mlir::Value ivSlot) {
  mlir::Region &cond = forOp.getCond();
  if (!cond.hasOneBlock())
    return mlir::failure();
  mlir::Block &condBlock = cond.front();
  auto condOp = mlir::dyn_cast<cir::ConditionOp>(condBlock.getTerminator());
  if (!condOp)
    return mlir::failure();
  // The compare must be rebuilt inside the condition region on every
  // iteration. One hoisted before the loop is evaluated only once.
  auto cmp = condOp.getCondition().getDefiningOp<cir::CmpOp>();
  if (!cmp || cmp->getBlock() != &condBlock)
    return mlir::failure();

  llvm::SmallPtrSet<mlir::Operation *, 8> lhsSlice, rhsSlice;
  collectSliceInBlock(cmp.getLhs(), condBlock, lhsSlice);
  collectSliceInBlock(cmp.getRhs(), condBlock, rhsSlice);
  bool lhsReadsIv = sliceReadsSlot(lhsSlice, ivSlot);
  bool rhsReadsIv = sliceReadsSlot(rhsSlice, ivSlot);
  // Exactly one side must vary with the induction variable.
  if (lhsReadsIv == rhsReadsIv)
    return mlir::failure();

  ControlComparison result;
  if (lhsReadsIv) {
    result = {cmp.getLhs(), cmp.getRhs(), cmp.getKind()};
  } else {
    result = {cmp.getRhs(), cmp.getLhs(), normalizeToIvOnLhs(cmp.getKind())};
  }

  // The region may compute the two compare operands and nothing else, using
  // only the supported vocabulary.
  for (mlir::Operation &op : condBlock) {
    if (&op == cmp.getOperation() || &op == condOp.getOperation())
      continue;
    if (!lhsSlice.contains(&op) && !rhsSlice.contains(&op))
      return mlir::failure();
    if (!isSupportedControlOp(&op))
      return mlir::failure();
  }
  return result;
}

mlir::FailureOr<CountedLoop> cir::loopopt::matchCountedLoop(cir::ForOp forOp) {
  // Match the frontend unit-step form exactly. The four operations are a
  // load, an increment or decrement, a store, and a yield.
  mlir::Region &step = forOp.getStep();
  if (!step.hasOneBlock() || step.front().getOperations().size() != 4)
    return mlir::failure();
  auto stepLoad = mlir::dyn_cast<cir::LoadOp>(&step.front().front());
  if (!stepLoad || !isOrdinaryAccess(stepLoad))
    return mlir::failure();
  mlir::Operation *update = stepLoad->getNextNode();
  if (!mlir::isa<cir::IncOp, cir::DecOp>(update) ||
      update->getOperand(0) != stepLoad.getResult())
    return mlir::failure();
  StepDirection direction = mlir::isa<cir::IncOp>(update)
                                ? StepDirection::Increment
                                : StepDirection::Decrement;
  auto stepStore = mlir::dyn_cast<cir::StoreOp>(update->getNextNode());
  if (!stepStore || !isOrdinaryAccess(stepStore) ||
      stepStore.getValue() != update->getResult(0) ||
      stepStore.getAddr() != stepLoad.getAddr())
    return mlir::failure();
  auto stepYield = mlir::dyn_cast<cir::YieldOp>(stepStore->getNextNode());
  if (!stepYield || stepYield->getNumOperands() != 0)
    return mlir::failure();
  mlir::Value ivSlot = stepStore.getAddr();
  if (!ivSlot.getDefiningOp<cir::AllocaOp>())
    return mlir::failure();

  mlir::FailureOr<ControlComparison> condition =
      matchControlComparison(forOp, ivSlot);
  if (mlir::failed(condition))
    return mlir::failure();

  if (!slotWritesAreOnlyTheStep(ivSlot, stepStore, forOp))
    return mlir::failure();

  cir::StoreOp initStore = getInitStore(ivSlot, forOp);
  if (!initStore)
    return mlir::failure();

  return CountedLoop{forOp, ivSlot, initStore.getValue(), *condition,
                     direction};
}

cir::ForOp cir::loopopt::getSingleInnerFor(cir::ForOp forOp) {
  cir::ForOp inner;
  bool multiple = false;
  // Ancestry is checked against any loop op, so a for wrapped in scopes is
  // found and a for inside a nested while is not claimed. The walk never
  // visits forOp itself.
  forOp.getBody().walk([&](cir::ForOp candidate) {
    if (candidate->getParentOfType<cir::LoopOpInterface>() !=
        mlir::cast<cir::LoopOpInterface>(forOp.getOperation()))
      return;
    if (inner)
      multiple = true;
    inner = candidate;
  });
  return multiple ? cir::ForOp{} : inner;
}
