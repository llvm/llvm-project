//====- LowerCIRLoopToSCF.cpp - Lowering from CIR Loop to SCF -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR loop operations to SCF.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/LowerToMLIR.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace cir;
using namespace llvm;

namespace cir {

class SCFLoop {
public:
  SCFLoop(mlir::cir::ForOp op, mlir::ConversionPatternRewriter *rewriter)
      : forOp(op), rewriter(rewriter) {}

  int64_t getStep() { return step; }
  mlir::Value getLowerBound() { return lowerBound; }
  mlir::Value getUpperBound() { return upperBound; }

  int64_t findStepAndIV(mlir::Value &addr);
  mlir::cir::CmpOp findCmpOp();
  mlir::Value findIVInitValue();
  void analysis();

  mlir::Value plusConstant(mlir::Value V, mlir::Location loc, int addend);
  void transferToSCFForOp();

private:
  mlir::cir::ForOp forOp;
  mlir::cir::CmpOp cmpOp;
  mlir::Value IVAddr, lowerBound = nullptr, upperBound = nullptr;
  mlir::ConversionPatternRewriter *rewriter;
  int64_t step = 0;
};

class SCFWhileLoop {
public:
  SCFWhileLoop(mlir::cir::WhileOp op, mlir::cir::WhileOp::Adaptor adaptor,
               mlir::ConversionPatternRewriter *rewriter)
      : whileOp(op), adaptor(adaptor), rewriter(rewriter) {}
  void transferToSCFWhileOp();

private:
  mlir::cir::WhileOp whileOp;
  mlir::cir::WhileOp::Adaptor adaptor;
  mlir::ConversionPatternRewriter *rewriter;
};

static int64_t getConstant(mlir::cir::ConstantOp op) {
  auto attr = op->getAttrs().front().getValue();
  const auto IntAttr = mlir::dyn_cast<mlir::cir::IntAttr>(attr);
  return IntAttr.getValue().getSExtValue();
}

int64_t SCFLoop::findStepAndIV(mlir::Value &addr) {
  auto *stepBlock =
      (forOp.maybeGetStep() ? &forOp.maybeGetStep()->front() : nullptr);
  assert(stepBlock && "Can not find step block");

  int64_t step = 0;
  mlir::Value IV = nullptr;
  // Try to match "IV load addr; ++IV; store IV, addr" to find step.
  for (mlir::Operation &op : *stepBlock)
    if (auto loadOp = dyn_cast<mlir::cir::LoadOp>(op)) {
      addr = loadOp.getAddr();
      IV = loadOp.getResult();
    } else if (auto cop = dyn_cast<mlir::cir::ConstantOp>(op)) {
      if (step)
        llvm_unreachable(
            "Not support multiple constant in step calculation yet");
      step = getConstant(cop);
    } else if (auto bop = dyn_cast<mlir::cir::BinOp>(op)) {
      if (bop.getLhs() != IV)
        llvm_unreachable("Find BinOp not operate on IV");
      if (bop.getKind() != mlir::cir::BinOpKind::Add)
        llvm_unreachable(
            "Not support BinOp other than Add in step calculation yet");
    } else if (auto uop = dyn_cast<mlir::cir::UnaryOp>(op)) {
      if (uop.getInput() != IV)
        llvm_unreachable("Find UnaryOp not operate on IV");
      if (uop.getKind() == mlir::cir::UnaryOpKind::Inc)
        step = 1;
      else if (uop.getKind() == mlir::cir::UnaryOpKind::Dec)
        llvm_unreachable("Not support decrement step yet");
    } else if (auto storeOp = dyn_cast<mlir::cir::StoreOp>(op)) {
      assert(storeOp.getAddr() == addr && "Can't find IV when lowering ForOp");
    }
  assert(step && "Can't find step when lowering ForOp");

  return step;
}

static bool isIVLoad(mlir::Operation *op, mlir::Value IVAddr) {
  if (!op)
    return false;
  if (isa<mlir::cir::LoadOp>(op)) {
    if (!op->getOperand(0))
      return false;
    if (op->getOperand(0) == IVAddr)
      return true;
  }
  return false;
}

mlir::cir::CmpOp SCFLoop::findCmpOp() {
  cmpOp = nullptr;
  for (auto *user : IVAddr.getUsers()) {
    if (user->getParentRegion() != &forOp.getCond())
      continue;
    if (auto loadOp = dyn_cast<mlir::cir::LoadOp>(*user)) {
      if (!loadOp->hasOneUse())
        continue;
      if (auto op = dyn_cast<mlir::cir::CmpOp>(*loadOp->user_begin())) {
        cmpOp = op;
        break;
      }
    }
  }
  if (!cmpOp)
    llvm_unreachable("Can't find loop CmpOp");

  auto type = cmpOp.getLhs().getType();
  if (!mlir::isa<mlir::cir::IntType>(type))
    llvm_unreachable("Non-integer type IV is not supported");

  auto lhsDefOp = cmpOp.getLhs().getDefiningOp();
  if (!lhsDefOp)
    llvm_unreachable("Can't find IV load");
  if (!isIVLoad(lhsDefOp, IVAddr))
    llvm_unreachable("cmpOp LHS is not IV");

  if (cmpOp.getKind() != mlir::cir::CmpOpKind::le &&
      cmpOp.getKind() != mlir::cir::CmpOpKind::lt)
    llvm_unreachable("Not support lowering other than le or lt comparison");

  return cmpOp;
}

mlir::Value SCFLoop::plusConstant(mlir::Value V, mlir::Location loc,
                                  int addend) {
  auto type = V.getType();
  auto c1 = rewriter->create<mlir::arith::ConstantOp>(
      loc, type, mlir::IntegerAttr::get(type, addend));
  return rewriter->create<mlir::arith::AddIOp>(loc, V, c1);
}

// Return IV initial value by searching the store before the loop.
// The operations before the loop have been transferred to MLIR.
// So we need to go through getRemappedValue to find the value.
mlir::Value SCFLoop::findIVInitValue() {
  auto remapAddr = rewriter->getRemappedValue(IVAddr);
  if (!remapAddr)
    return nullptr;
  if (!remapAddr.hasOneUse())
    return nullptr;
  auto memrefStore = dyn_cast<mlir::memref::StoreOp>(*remapAddr.user_begin());
  if (!memrefStore)
    return nullptr;
  return memrefStore->getOperand(0);
}

void SCFLoop::analysis() {
  step = findStepAndIV(IVAddr);
  cmpOp = findCmpOp();
  auto IVInit = findIVInitValue();
  // The loop end value should be hoisted out of loop by -cir-mlir-scf-prepare.
  // So we could get the value by getRemappedValue.
  auto IVEndBound = rewriter->getRemappedValue(cmpOp.getRhs());
  // If the loop end bound is not loop invariant and can't be hoisted.
  // The following assertion will be triggerred.
  assert(IVEndBound && "can't find IV end boundary");

  if (step > 0) {
    lowerBound = IVInit;
    if (cmpOp.getKind() == mlir::cir::CmpOpKind::lt)
      upperBound = IVEndBound;
    else if (cmpOp.getKind() == mlir::cir::CmpOpKind::le)
      upperBound = plusConstant(IVEndBound, cmpOp.getLoc(), 1);
  }
  assert(lowerBound && "can't find loop lower bound");
  assert(upperBound && "can't find loop upper bound");
}

// Return true if op operation is in the loop body.
static bool isInLoopBody(mlir::Operation *op) {
  mlir::Operation *parentOp = op->getParentOp();
  if (!parentOp)
    return false;
  if (isa<mlir::scf::ForOp>(parentOp))
    return true;
  auto forOp = dyn_cast<mlir::cir::ForOp>(parentOp);
  if (forOp && (&forOp.getBody() == op->getParentRegion()))
    return true;
  return false;
}

void SCFLoop::transferToSCFForOp() {
  auto ub = getUpperBound();
  auto lb = getLowerBound();
  auto loc = forOp.getLoc();
  auto type = lb.getType();
  auto step = rewriter->create<mlir::arith::ConstantOp>(
      loc, type, mlir::IntegerAttr::get(type, getStep()));
  auto scfForOp = rewriter->create<mlir::scf::ForOp>(loc, lb, ub, step);
  SmallVector<mlir::Value> bbArg;
  rewriter->eraseOp(&scfForOp.getBody()->back());
  rewriter->inlineBlockBefore(&forOp.getBody().front(), scfForOp.getBody(),
                              scfForOp.getBody()->end(), bbArg);
  scfForOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<mlir::cir::BreakOp>(op) || isa<mlir::cir::ContinueOp>(op) ||
        isa<mlir::cir::IfOp>(op))
      llvm_unreachable(
          "Not support lowering loop with break, continue or if yet");
    // Replace the IV usage to scf loop induction variable.
    if (isIVLoad(op, IVAddr)) {
      auto newIV = scfForOp.getInductionVar();
      op->getResult(0).replaceAllUsesWith(newIV);
      // Only erase the IV load in the loop body because all the operations
      // in loop step and condition regions will be erased.
      if (isInLoopBody(op))
        rewriter->eraseOp(op);
    }
    return mlir::WalkResult::advance();
  });
}

void SCFWhileLoop::transferToSCFWhileOp() {
  auto scfWhileOp = rewriter->create<mlir::scf::WhileOp>(
      whileOp->getLoc(), whileOp->getResultTypes(), adaptor.getOperands());
  rewriter->createBlock(&scfWhileOp.getBefore());
  rewriter->createBlock(&scfWhileOp.getAfter());

  rewriter->cloneRegionBefore(whileOp.getCond(),
                              &scfWhileOp.getBefore().back());
  rewriter->eraseBlock(&scfWhileOp.getBefore().back());

  rewriter->cloneRegionBefore(whileOp.getBody(), &scfWhileOp.getAfter().back());
  rewriter->eraseBlock(&scfWhileOp.getAfter().back());
}

class CIRForOpLowering : public mlir::OpConversionPattern<mlir::cir::ForOp> {
public:
  using OpConversionPattern<mlir::cir::ForOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ForOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SCFLoop loop(op, &rewriter);
    loop.analysis();
    loop.transferToSCFForOp();
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CIRWhileOpLowering
    : public mlir::OpConversionPattern<mlir::cir::WhileOp> {
public:
  using OpConversionPattern<mlir::cir::WhileOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::WhileOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SCFWhileLoop loop(op, adaptor, &rewriter);
    loop.transferToSCFWhileOp();
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CIRConditionOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ConditionOp> {
public:
  using OpConversionPattern<mlir::cir::ConditionOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ConditionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *parentOp = op->getParentOp();
    return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(parentOp)
        .Case<mlir::scf::WhileOp>([&](auto) {
          auto condition = adaptor.getCondition();
          auto i1Condition = rewriter.create<mlir::arith::TruncIOp>(
              op.getLoc(), rewriter.getI1Type(), condition);
          rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
              op, i1Condition, parentOp->getOperands());
          return mlir::success();
        })
        .Default([](auto) { return mlir::failure(); });
  }
};

void populateCIRLoopToSCFConversionPatterns(mlir::RewritePatternSet &patterns,
                                            mlir::TypeConverter &converter) {
  patterns.add<CIRForOpLowering, CIRWhileOpLowering, CIRConditionOpLowering>(
      converter, patterns.getContext());
}

} // namespace cir