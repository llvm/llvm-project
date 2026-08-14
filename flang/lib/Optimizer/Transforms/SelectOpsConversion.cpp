//===- SelectOpsConversion.cpp - Lower fir.select* to cf.* ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"

namespace fir {
#define GEN_PASS_DEF_SELECTOPSCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "fir-select-ops-conversion"

namespace {
using namespace mlir;

template <typename SwitchLike>
static ValueRange successorOperands(SwitchLike op, unsigned successorIdx) {
  return op.getSuccessorOperands(successorIdx).getForwardedOperands();
}

// Bit-cast a signed or unsigned FIR integer value to its signless-integer
// equivalent. The arith / cf dialects only accept signless integers, so any
// `ui*` / `si*` value must be normalized before use.
static Value toSignlessInteger(OpBuilder &b, Location loc, Value v) {
  auto intTy = dyn_cast<IntegerType>(v.getType());
  if (!intTy || intTy.isSignless())
    return v;
  auto signlessTy = IntegerType::get(v.getContext(), intTy.getWidth());
  return fir::ConvertOp::create(b, loc, signlessTy, v);
}

// Widen `v` to a signless i64, the type used for `cf.switch` selectors in
// this pass. `v` must be integer- or index-typed. Wider-than-64-bit
// selectors are truncated to i64, matching the behaviour of the FIR-to-LLVM
// `integerCast` helper this pass replaces.
static Value toSignlessI64(OpBuilder &b, Location loc, Value v) {
  v = toSignlessInteger(b, loc, v);
  auto i64 = IntegerType::get(v.getContext(), 64);
  Type srcTy = v.getType();
  if (srcTy == i64)
    return v;
  if (isa<IndexType>(srcTy))
    return arith::IndexCastOp::create(b, loc, i64, v);
  unsigned srcW = cast<IntegerType>(srcTy).getWidth();
  if (srcW < 64)
    return arith::ExtSIOp::create(b, loc, i64, v);
  return arith::TruncIOp::create(b, loc, i64, v);
}

//===----------------------------------------------------------------------===//
// fir.select / fir.select_rank → cf.switch
//===----------------------------------------------------------------------===//

template <typename SwitchLike>
static LogicalResult lowerToSwitch(SwitchLike op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // The fir.select* selector must be integer or index typed. (fir.select_rank
  // in particular reaches this pass with an integer selector: flang emits a
  // `fir.box_rank` earlier that reads the rank field from the CFI descriptor
  // as an `i8`, and the `fir.select_rank` then dispatches on that.)
  Type selectorTy = op.getSelector().getType();
  if (!isa<IntegerType, IndexType>(selectorTy))
    return op.emitOpError("selector is not an integer/index type");

  // Widen the selector to i64. `cf.switch`'s case-value APInts are also
  // constructed at 64 bits below, so the case-value / selector element-type
  // constraint is satisfied. Widening to i64 additionally accommodates
  // Fortran computed GOTOs whose label values may not fit in the source
  // selector's width.
  Value selector = toSignlessI64(builder, loc, op.getSelector());

  SmallVector<int64_t> caseValues;
  SmallVector<Block *> caseDests;
  SmallVector<ValueRange> caseOperands;
  Block *defaultDest = nullptr;
  ValueRange defaultOperands;

  unsigned numConds = op.getNumConditions();
  ArrayRef<Attribute> cases = op.getCases().getValue();
  for (unsigned i = 0; i != numConds; ++i) {
    Block *dest = op.getSuccessor(i);
    ValueRange ops = successorOperands(op, i);
    Attribute attr = cases[i];

    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      caseValues.push_back(intAttr.getInt());
      caseDests.push_back(dest);
      caseOperands.push_back(ops);
      continue;
    }

    assert(isa<UnitAttr>(attr) && "unexpected case attribute kind");
    assert(!defaultDest && "multiple unit (default) entries in fir.select*");
    defaultDest = dest;
    defaultOperands = ops;
  }

  if (!defaultDest)
    return op.emitOpError("fir.select* verifier requires a unit default, "
                          "but none was found");

  if (caseValues.empty()) {
    cf::BranchOp::create(builder, loc, defaultDest, defaultOperands);
  } else {
    SmallVector<APInt> caseAPInts;
    caseAPInts.reserve(caseValues.size());
    for (int64_t v : caseValues)
      caseAPInts.emplace_back(64, v, /*isSigned=*/true);
    cf::SwitchOp::create(builder, loc, selector, defaultDest, defaultOperands,
                         caseAPInts, caseDests, caseOperands);
  }

  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// fir.select_case → if-then-else ladder of cf.cond_br
//===----------------------------------------------------------------------===//

// Emit one rung of the fir.select_case if-then-else ladder:
//
//   thisBlock:  ...; cf.cond_br %cmp, dest(destOps), nextBlock
//   nextBlock:  <insertion point at end>
//
// Returns the freshly-created nextBlock.
static Block *genCaseLadderStep(OpBuilder &builder, Location loc, Value cmp,
                                Block *dest, ValueRange destOps) {
  Block *thisBlock = builder.getInsertionBlock();
  Region *region = thisBlock->getParent();
  Block *nextBlock =
      builder.createBlock(region, std::next(thisBlock->getIterator()));
  builder.setInsertionPointToEnd(thisBlock);
  cf::CondBranchOp::create(builder, loc, cmp, dest, destOps, nextBlock,
                           ValueRange());
  builder.setInsertionPointToEnd(nextBlock);
  return nextBlock;
}

static LogicalResult lowerSelectCase(fir::SelectCaseOp op) {
  Location loc = op.getLoc();
  Value selector = op.getSelector();

  // CHARACTER selectors are not yet supported.
  if (isa<fir::CharacterType>(selector.getType())) {
    TODO(op.getLoc(), "fir.select_case codegen with character type");
    return failure();
  }

  if (!isa<IntegerType, IndexType>(selector.getType()))
    return op.emitOpError("non-integer/character selector not supported");

  OpBuilder builder(op);
  // Fortran `UNSIGNED` selectors have `ui*` type; convert to signless up
  // front and pick the unsigned compare predicate for range checks. Signed
  // and signless selectors use the signed predicate.
  auto origSelTy = dyn_cast<IntegerType>(selector.getType());
  bool isUnsigned = origSelTy && origSelTy.isUnsigned();
  arith::CmpIPredicate lePred =
      isUnsigned ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle;
  selector = toSignlessInteger(builder, loc, selector);
  unsigned numConds = op.getNumConditions();
  ArrayRef<Attribute> cases = op.getCases().getValue();

  for (unsigned i = 0; i != numConds; ++i) {
    Block *dest = op.getSuccessor(i);
    ValueRange destOps = successorOperands(op, i);
    Attribute attr = cases[i];

    if (isa<UnitAttr>(attr)) {
      // Default branch — unconditional jump. Must be the last entry.
      assert(i + 1 == numConds && "fir.select_case unit attr must be last");
      cf::BranchOp::create(builder, loc, dest, destOps);
      op.erase();
      return success();
    }

    std::optional<OperandRange> cmpOpsOpt = op.getCompareOperands(i);
    if (!cmpOpsOpt)
      return op.emitOpError("missing compare operands for case ") << i;
    OperandRange cmpOps = *cmpOpsOpt;

    if (isa<fir::PointIntervalAttr>(attr)) {
      Value cmpOp = toSignlessInteger(builder, loc, cmpOps.front());
      Value cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                        selector, cmpOp);
      genCaseLadderStep(builder, loc, cmp, dest, destOps);
      continue;
    }
    if (isa<fir::LowerBoundAttr>(attr)) {
      // case(c:): match when c <= selector.
      Value cmpOp = toSignlessInteger(builder, loc, cmpOps.front());
      Value cmp = arith::CmpIOp::create(builder, loc, lePred, cmpOp, selector);
      genCaseLadderStep(builder, loc, cmp, dest, destOps);
      continue;
    }
    if (isa<fir::UpperBoundAttr>(attr)) {
      // case(:c): match when selector <= c.
      Value cmpOp = toSignlessInteger(builder, loc, cmpOps.front());
      Value cmp = arith::CmpIOp::create(builder, loc, lePred, selector, cmpOp);
      genCaseLadderStep(builder, loc, cmp, dest, destOps);
      continue;
    }
    if (isa<fir::ClosedIntervalAttr>(attr)) {
      // case(lo:hi): two-step short-circuit. First check lo <= selector;
      // if true, branch to a hi-check block; if false, fall through.
      Value lo = toSignlessInteger(builder, loc, cmpOps[0]);
      Value hi = toSignlessInteger(builder, loc, cmpOps[1]);
      Value cmpLo = arith::CmpIOp::create(builder, loc, lePred, lo, selector);
      // Block layout (in source order):
      //   thisBlock     -> cond_br cmpLo, hiCheck, fallThrough
      //   hiCheck       -> cond_br cmpHi, dest, fallThrough
      //   fallThrough   -> next case
      Block *thisBlock = builder.getInsertionBlock();
      Region *region = thisBlock->getParent();
      auto insertIt = std::next(thisBlock->getIterator());
      Block *hiCheck = builder.createBlock(region, insertIt);
      Block *fallThrough =
          builder.createBlock(region, std::next(hiCheck->getIterator()));
      builder.setInsertionPointToEnd(thisBlock);
      cf::CondBranchOp::create(builder, loc, cmpLo, hiCheck, ValueRange(),
                               fallThrough, ValueRange());
      builder.setInsertionPointToEnd(hiCheck);
      Value cmpHi = arith::CmpIOp::create(builder, loc, lePred, selector, hi);
      cf::CondBranchOp::create(builder, loc, cmpHi, dest, destOps, fallThrough,
                               ValueRange());
      builder.setInsertionPointToEnd(fallThrough);
      continue;
    }
    return op.emitOpError("unknown case attribute kind");
  }

  // The FIR verifier requires a `unit` entry, so we should not reach here.
  return op.emitOpError("fir.select_case has no unit (default) entry");
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class SelectOpsConversion
    : public fir::impl::SelectOpsConversionBase<SelectOpsConversion> {
public:
  using SelectOpsConversionBase<SelectOpsConversion>::SelectOpsConversionBase;

  void runOnOperation() override {
    // Collect first; rewriting mutates blocks and would invalidate a live walk.
    SmallVector<Operation *> worklist;
    getOperation()->walk([&](Operation *op) {
      if (isa<fir::SelectOp, fir::SelectCaseOp, fir::SelectRankOp>(op))
        worklist.push_back(op);
    });

    for (Operation *op : worklist) {
      LogicalResult r = success();
      if (auto s = dyn_cast<fir::SelectOp>(op))
        r = lowerToSwitch(s);
      else if (auto sr = dyn_cast<fir::SelectRankOp>(op))
        r = lowerToSwitch(sr);
      else if (auto sc = dyn_cast<fir::SelectCaseOp>(op))
        r = lowerSelectCase(sc);
      if (failed(r)) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
