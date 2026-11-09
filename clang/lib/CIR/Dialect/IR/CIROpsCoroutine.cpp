//===- CIROpsCoroutine.cpp - CIR ops in partition `Coroutine` -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR ops in partition `Coroutine`: the hand-written
// parts first, then the TableGen-generated definitions.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIROpsCoroutine.h"
#include "clang/CIR/Dialect/IR/CIROpsControlFlow.h"

#include "CIROpDefinitions.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// CoroSuspendPoint
//===----------------------------------------------------------------------===//

MutableOperandRange
cir::CoroSuspendPoint::getMutableSuccessorOperands(RegionSuccessor point) {
  // The condition operand is not forwarded to successor regions.
  return MutableOperandRange(getOperation(), 0, 0);
}

//===----------------------------------------------------------------------===//
// AwaitOp
//===----------------------------------------------------------------------===//

void cir::AwaitOp::build(OpBuilder &builder, OperationState &result,
                         cir::AwaitKind kind, BuilderCallbackRef readyBuilder,
                         BuilderCallbackRef suspendBuilder,
                         BuilderCallbackRef resumeBuilder) {
  result.addAttribute(getKindAttrName(result.name),
                      cir::AwaitKindAttr::get(builder.getContext(), kind));
  {
    OpBuilder::InsertionGuard guard(builder);
    Region *readyRegion = result.addRegion();
    builder.createBlock(readyRegion);
    readyBuilder(builder, result.location);
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    Region *suspendRegion = result.addRegion();
    builder.createBlock(suspendRegion);
    suspendBuilder(builder, result.location);
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    Region *resumeRegion = result.addRegion();
    builder.createBlock(resumeRegion);
    resumeBuilder(builder, result.location);
  }
}

void cir::AwaitOp::getSuccessorRegions(
    mlir::RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  assert(point.isParent() || point.getTerminatorPredecessorOrNull());

  // Execution always starts in the ready region.
  if (point.isParent()) {
    regions.emplace_back(&getReady());
    return;
  }

  mlir::Region *parentRegion =
      point.getTerminatorPredecessorOrNull()->getParentRegion();

  // Branching from ready: the cir.condition terminating it selects between
  // suspending and resuming. Keep in sync with
  // ConditionOp::getSuccessorRegions.
  //
  // TODO: retrieve information from the promise and only push the
  // necessary ones. Example: `std::suspend_never` on initial or final
  // await's might allow suspend region to be skipped.
  if (&getReady() == parentRegion) {
    regions.emplace_back(&getResume());
    regions.emplace_back(&getSuspend());
    return;
  }

  // Branching from suspend: continue in resume once the coroutine is resumed
  // (or immediately if suspension is vetoed), or exit to the parent operation
  // if it stays suspended or is destroyed.
  if (&getSuspend() == parentRegion) {
    regions.emplace_back(&getResume());
    regions.emplace_back(getOperation());
    return;
  }

  // Branching from resume: exit to the parent operation.
  regions.emplace_back(getOperation());
}

LogicalResult cir::AwaitOp::verify() {
  if (!isa<ConditionOp>(this->getReady().back().getTerminator()))
    return emitOpError("ready region must end with cir.condition");
  if (this->getSuspend().empty())
    return emitOpError("suspend region must not be empty");
  if (!isa<CoroSuspendPoint>(this->getSuspend().back().getTerminator()))
    return emitOpError("suspend region must end with cir.coro.suspend_point");
  return success();
}

//===----------------------------------------------------------------------===//
// CoReturnOp
//===----------------------------------------------------------------------===//

LogicalResult cir::CoReturnOp::verify() {
  mlir::Operation *coRet = getOperation();
  auto coroutine = coRet->getParentOfType<CoroutineOp>();
  if (!coroutine.getBody().isAncestor(getOperation()->getParentRegion()))
    return emitOpError("must be inside the cir.coroutine body region");
  return success();
}

//===----------------------------------------------------------------------===//
// CoroutineOp
//===----------------------------------------------------------------------===//

void cir::CoroutineOp::build(OpBuilder &builder, OperationState &result,
                             BuilderCallbackRef initialSuspendBuilder,
                             BuilderCallbackRef bodyBuilder,
                             BuilderCallbackRef finalSuspendBuilder,
                             BuilderCallbackRef destroyBuilder,
                             BuilderCallbackRef exitBuilder) {
  {
    OpBuilder::InsertionGuard guard(builder);
    Region *initialRegion = result.addRegion();
    builder.createBlock(initialRegion);
    initialSuspendBuilder(builder, result.location);
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    Region *bodyRegion = result.addRegion();
    builder.createBlock(bodyRegion);
    bodyBuilder(builder, result.location);
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    Region *finalRegion = result.addRegion();
    builder.createBlock(finalRegion);
    finalSuspendBuilder(builder, result.location);
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    Region *destroyRegion = result.addRegion();
    builder.createBlock(destroyRegion);
    destroyBuilder(builder, result.location);
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    Region *exitRegion = result.addRegion();
    builder.createBlock(exitRegion);
    exitBuilder(builder, result.location);
  }
}

void cir::CoroutineOp::getSuccessorRegions(
    mlir::RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  // The parent op always enters through initial_suspend get_return_object()
  // (if present) and the init await always run first.
  if (point.isParent()) {
    regions.emplace_back(&getInitialSuspend());
    return;
  }

  mlir::Region *parent =
      point.getTerminatorPredecessorOrNull()->getParentRegion();

  // initial_suspend either falls into body (resumed, or never actually
  // suspended because await_ready() was true), exits directly, or reaches
  // destroy when the coroutine is destroyed while suspended at its initial
  // suspend point.
  if (parent == &getInitialSuspend()) {
    regions.emplace_back(&getBody());
    regions.emplace_back(&getExit());
    regions.emplace_back(&getDestroy());
    return;
  }

  // body can fall through into final_suspend (co_return), exit directly
  // from any suspend_point inside it that just suspends, or reach destroy,
  // either from a suspend_point being destroyed, or from an exception
  // escaping body's own catch-all.
  if (parent == &getBody()) {
    regions.emplace_back(&getFinalSuspend());
    regions.emplace_back(&getExit());    // any suspend_point inside body
    regions.emplace_back(&getDestroy()); // destroy() on any suspend inside body
    return;
  }

  // final_suspend's only live edge in valid programs is destroy. Resuming
  // past the final suspend is UB. The ready-immediately edge to exit is
  // kept for structural symmetry with the other two suspend regions even
  // though it's effectively dead.
  if (parent == &getFinalSuspend()) {
    regions.emplace_back(
        &getDestroy());               // the only real exit from final_suspend
    regions.emplace_back(&getExit()); // ready==true edge, rarely taken
    return;
  }

  // destroy has two possible outcomes depending on why it was entered:
  // ordinary destroy dispatch falls through to exit (normal return); an
  // exception that reached destroy needs to keep propagating instead, i.e.
  // leave the whole op rather than go through exit's cir.return.
  if (parent == &getDestroy()) {
    regions.emplace_back(&getExit());
    regions.push_back(RegionSuccessor(getOperation())); // unwind out of the op
    return;
  }
  // exit always terminates the op there's no region it loops back into.
  if (parent == &getExit()) {
    regions.emplace_back(getOperation());
    return;
  }
}

mlir::ValueRange
cir::CoroutineOp::getSuccessorInputs(RegionSuccessor successor) {
  return ValueRange();
}

LogicalResult cir::CoroutineOp::verify() {
  mlir::Region &initialSuspend = getInitialSuspend();
  mlir::Region &body = getBody();
  mlir::Region &finalSuspend = getFinalSuspend();
  mlir::Region &destroy = getDestroy();
  mlir::Region &exit = getExit();

  // initial_suspend must contain exactly one 'init'-kind cir.await an
  // optional prelude of ordinary statements (get_return_object() and
  // similar) is allowed before it, but no *other* await may appear.
  int initAwaitCount = 0;
  bool hasInvalidAwait = false;

  initialSuspend.walk([&](Operation *op) {
    auto awaitOp = mlir::dyn_cast<AwaitOp>(op);
    if (!awaitOp)
      return mlir::WalkResult::advance();

    if (awaitOp.getKind() == cir::AwaitKind::Init)
      ++initAwaitCount;
    else
      hasInvalidAwait = true;

    if (initAwaitCount > 1 || hasInvalidAwait)
      return mlir::WalkResult::interrupt();

    return mlir::WalkResult::advance();
  });

  if (hasInvalidAwait)
    return emitOpError("'initial_suspend' must not contain any cir.await other "
                       "than the 'init' one");
  if (initAwaitCount != 1)
    return emitOpError(
        "must have exactly one 'init' cir.await in 'initial_suspend'");

  // final_suspend must contain only one or zero 'final'-kind cir.await
  int finalAwaitCount = 0;
  hasInvalidAwait = false;

  finalSuspend.walk([&](Operation *op) {
    auto awaitOp = mlir::dyn_cast<AwaitOp>(op);
    if (!awaitOp)
      return mlir::WalkResult::advance();

    if (awaitOp.getKind() == cir::AwaitKind::Final)
      ++finalAwaitCount;
    else
      hasInvalidAwait = true;

    if (finalAwaitCount > 1 || hasInvalidAwait)
      return mlir::WalkResult::interrupt();

    return mlir::WalkResult::advance();
  });

  if (hasInvalidAwait)
    return emitOpError("'final_suspend' must not contain any cir.await other "
                       "than the 'final' one");
  if (finalAwaitCount > 1)
    return emitOpError(
        "must have one or zero 'final' cir.await in 'final_suspend'");

  // Each region must end with the terminator its role requires:
  // initial_suspend/final_suspend/destroy yield back into cir.coroutine's
  // own control flow, body ends in a cir.co_return (or a plain yield if
  // some path never reaches one), and exit actually returns from the
  // function.
  if (initialSuspend.empty() || initialSuspend.back().empty())
    return emitOpError("initial_suspend region must not be empty");
  if (!isa<YieldOp>(initialSuspend.back().back()))
    return emitOpError("'initial_suspend' must end with cir.yield");

  if (body.empty() || body.back().empty())
    return emitOpError("'body' region must not be empty");
  if (!isa<YieldOp, CoReturnOp>(body.back().back()))
    return emitOpError("'body' must end with cir.yield or cir.co_return");

  if (finalSuspend.empty() || finalSuspend.back().empty())
    return emitOpError("'final_suspend' region must not be empty");
  if (!isa<YieldOp>(finalSuspend.back().back()))
    return emitOpError("'final_suspend' must end with cir.yield");

  if (destroy.empty() || destroy.back().empty())
    return emitOpError("'destroy' region must not be empty");
  if (!isa<YieldOp>(destroy.back().back()))
    return emitOpError("'destroy' must end with cir.yield");

  if (exit.empty() || exit.back().empty())
    return emitOpError("'exit' region must not be empty");
  if (!isa<ReturnOp>(exit.back().back()))
    return emitOpError("'exit' must end with cir.return");

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsCoroutine.cpp.inc"
