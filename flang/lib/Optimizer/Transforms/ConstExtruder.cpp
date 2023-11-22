//===- ConstExtruder.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include <atomic>

namespace fir {
#define GEN_PASS_DEF_CONSTEXTRUDEROPT
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-const-extruder-opt"

namespace {
std::atomic<int> uniqueLitId = 1;

static bool needsExtrusion(const mlir::Value *a) {
  if (!a || !a->getDefiningOp())
    return false;

  // is alloca
  if (auto alloca = mlir::dyn_cast_or_null<fir::AllocaOp>(a->getDefiningOp())) {
    // alloca has annotation
    if (alloca->hasAttr(fir::getAdaptToByRefAttrName())) {
      for (mlir::Operation *s : alloca.getOperation()->getUsers()) {
        if (const auto store = mlir::dyn_cast_or_null<fir::StoreOp>(s)) {
          auto constant_def = store->getOperand(0).getDefiningOp();
          // Expect constant definition operation
          if (mlir::isa<mlir::arith::ConstantOp>(constant_def)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

class CallOpRewriter : public mlir::OpRewritePattern<fir::CallOp> {
protected:
  mlir::DominanceInfo &di;

public:
  using OpRewritePattern::OpRewritePattern;

  CallOpRewriter(mlir::MLIRContext *ctx, mlir::DominanceInfo &_di)
      : OpRewritePattern(ctx), di(_di) {}

  mlir::LogicalResult
  matchAndRewrite(fir::CallOp callOp,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Processing call op: " << callOp << "\n");
    auto module = callOp->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, module);
    llvm::SmallVector<mlir::Value> newOperands;
    llvm::SmallVector<mlir::Operation *> toErase;
    for (const auto &a : callOp.getArgs()) {
      if (auto alloca =
              mlir::dyn_cast_or_null<fir::AllocaOp>(a.getDefiningOp())) {
        if (needsExtrusion(&a)) {

          mlir::Type varTy = alloca.getInType();
          assert(!fir::hasDynamicSize(varTy) &&
                 "only expect statically sized scalars to be by value");

          // find immediate store with const argument
          llvm::SmallVector<mlir::Operation *> stores;
          for (mlir::Operation *s : alloca.getOperation()->getUsers())
            if (mlir::isa<fir::StoreOp>(s) && di.dominates(s, callOp))
              stores.push_back(s);
          assert(stores.size() == 1 && "expected exactly one store");
          LLVM_DEBUG(llvm::dbgs() << " found store " << *stores[0] << "\n");

          auto constant_def = stores[0]->getOperand(0).getDefiningOp();
          // Expect constant definition operation or force legalisation of the
          // callOp and continue with its next argument
          if (!mlir::isa<mlir::arith::ConstantOp>(constant_def)) {
            // unable to remove alloca arg
            newOperands.push_back(a);
            continue;
          }

          LLVM_DEBUG(llvm::dbgs() << " found define " << *constant_def << "\n");

          auto loc = callOp.getLoc();
          llvm::StringRef globalPrefix = "_extruded_";

          std::string globalName;
          while (!globalName.length() || builder.getNamedGlobal(globalName))
            globalName =
                globalPrefix.str() + "." + std::to_string(uniqueLitId++);

          if (alloca->hasOneUse()) {
            toErase.push_back(alloca);
            toErase.push_back(stores[0]);
          } else {
            int count = -2;
            for (mlir::Operation *s : alloca.getOperation()->getUsers())
              if (di.dominates(stores[0], s))
                ++count;

            // delete if dominates itself and one more operation (which should
            // be callOp)
            if (!count)
              toErase.push_back(stores[0]);
          }
          auto global = builder.createGlobalConstant(
              loc, varTy, globalName,
              [&](fir::FirOpBuilder &builder) {
                mlir::Operation *cln = constant_def->clone();
                builder.insert(cln);
                fir::ExtendedValue exv{cln->getResult(0)};
                mlir::Value valBase = fir::getBase(exv);
                mlir::Value val = builder.createConvert(loc, varTy, valBase);
                builder.create<fir::HasValueOp>(loc, val);
              },
              builder.createInternalLinkage());
          mlir::Value ope = {builder.create<fir::AddrOfOp>(
              loc, global.resultType(), global.getSymbol())};
          newOperands.push_back(ope);
        } else {
          // alloca but without attr, add it
          newOperands.push_back(a);
        }
      } else {
        // non-alloca operand, add it
        newOperands.push_back(a);
      }
    }

    auto loc = callOp.getLoc();
    llvm::SmallVector<mlir::Type> newResultTypes;
    newResultTypes.append(callOp.getResultTypes().begin(),
                          callOp.getResultTypes().end());
    fir::CallOp newOp = builder.create<fir::CallOp>(
        loc, newResultTypes,
        callOp.getCallee().has_value() ? callOp.getCallee().value()
                                       : mlir::SymbolRefAttr{},
        newOperands, callOp.getFastmathAttr());
    rewriter.replaceOp(callOp, newOp);

    for (auto e : toErase)
      rewriter.eraseOp(e);

    LLVM_DEBUG(llvm::dbgs() << "extruded constant for " << callOp << " as "
                            << newOp << '\n');
    return mlir::success();
  }
};

// This pass attempts to convert immediate scalar literals in function calls
// to global constants to allow transformations as Dead Argument Elimination
class ConstExtruderOpt
    : public fir::impl::ConstExtruderOptBase<ConstExtruderOpt> {
protected:
  mlir::DominanceInfo *di;

public:
  ConstExtruderOpt() {}

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    di = &getAnalysis<mlir::DominanceInfo>();
    mod.walk([this](mlir::func::FuncOp func) { runOnFunc(func); });
  }

  void runOnFunc(mlir::func::FuncOp &func) {
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    // If func is a declaration, skip it.
    if (func.empty())
      return;

    target.addLegalDialect<fir::FIROpsDialect, mlir::arith::ArithDialect,
                           mlir::func::FuncDialect>();
    target.addDynamicallyLegalOp<fir::CallOp>([&](fir::CallOp op) {
      for (auto a : op.getArgs()) {
        if (needsExtrusion(&a))
          return false;
      }
      return true;
    });

    patterns.insert<CallOpRewriter>(context, *di);
    if (mlir::failed(
            mlir::applyPartialConversion(func, target, std::move(patterns)))) {
      mlir::emitError(func.getLoc(),
                      "error in constant extrusion optimization\n");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> fir::createConstExtruderPass() {
  return std::make_unique<ConstExtruderOpt>();
}
