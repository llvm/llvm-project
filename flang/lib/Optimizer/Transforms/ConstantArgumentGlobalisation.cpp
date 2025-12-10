//===- ConstantArgumentGlobalisation.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_CONSTANTARGUMENTGLOBALISATIONOPT
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-constant-argument-globalisation-opt"

namespace {
unsigned uniqueLitId = 1;

class CallOpRewriter : public mlir::OpRewritePattern<fir::CallOp> {
protected:
  const mlir::DominanceInfo &di;

public:
  using OpRewritePattern::OpRewritePattern;

  CallOpRewriter(mlir::MLIRContext *ctx, const mlir::DominanceInfo &_di)
      : OpRewritePattern(ctx), di(_di) {}

  llvm::LogicalResult
  matchAndRewrite(fir::CallOp callOp,
                  mlir::PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Processing call op: " << callOp << "\n");
    auto module = callOp->getParentOfType<mlir::ModuleOp>();
    bool needUpdate = false;
    fir::FirOpBuilder builder(rewriter, module);
    llvm::SmallVector<mlir::Value> newOperands;
    llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>> allocas;
    for (const mlir::Value &a : callOp.getArgs()) {
      auto alloca = mlir::dyn_cast_or_null<fir::AllocaOp>(a.getDefiningOp());
      // We can convert arguments that are alloca, and that has
      // the value by reference attribute. All else is just added
      // to the argument list.
      if (!alloca || !alloca->hasAttr(fir::getAdaptToByRefAttrName())) {
        newOperands.push_back(a);
        continue;
      }

      mlir::Type varTy = alloca.getInType();
      assert(!fir::hasDynamicSize(varTy) &&
             "only expect statically sized scalars to be by value");

      // Find immediate store with const argument
      mlir::Operation *store = nullptr;
      for (mlir::Operation *s : alloca->getUsers()) {
        if (mlir::isa<fir::StoreOp>(s) && di.dominates(s, callOp)) {
          // We can only deal with ONE store - if already found one,
          // set to nullptr and exit the loop.
          if (store) {
            store = nullptr;
            break;
          }
          store = s;
        }
      }

      // If we didn't find any store, or multiple stores, add argument as is
      // and move on.
      if (!store) {
        newOperands.push_back(a);
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << " found store " << *store << "\n");

      mlir::Operation *definingOp = store->getOperand(0).getDefiningOp();
      // If not a constant, add to operands and move on.
      if (!mlir::isa<mlir::arith::ConstantOp>(definingOp)) {
        // Unable to remove alloca arg
        newOperands.push_back(a);
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << " found define " << *definingOp << "\n");

      std::string globalName =
          "_global_const_." + std::to_string(uniqueLitId++);
      assert(!builder.getNamedGlobal(globalName) &&
             "We should have a unique name here");

      if (llvm::none_of(allocas,
                        [alloca](auto x) { return x.first == alloca; })) {
        allocas.push_back(std::make_pair(alloca, store));
      }

      auto loc = callOp.getLoc();
      fir::GlobalOp global = builder.createGlobalConstant(
          loc, varTy, globalName,
          [&](fir::FirOpBuilder &builder) {
            mlir::Operation *cln = definingOp->clone();
            builder.insert(cln);
            mlir::Value val =
                builder.createConvert(loc, varTy, cln->getResult(0));
            fir::HasValueOp::create(builder, loc, val);
          },
          builder.createInternalLinkage());
      mlir::Value addr = fir::AddrOfOp::create(
          builder, loc, global.resultType(), global.getSymbol());
      newOperands.push_back(addr);
      needUpdate = true;
    }

    if (needUpdate) {
      auto loc = callOp.getLoc();
      llvm::SmallVector<mlir::Type> newResultTypes;
      newResultTypes.append(callOp.getResultTypes().begin(),
                            callOp.getResultTypes().end());
      fir::CallOp newOp = fir::CallOp::create(builder, loc,
                                              callOp.getCallee().has_value()
                                                  ? callOp.getCallee().value()
                                                  : mlir::SymbolRefAttr{},
                                              newResultTypes, newOperands);
      // Copy all the attributes from the old to new op.
      newOp->setAttrs(callOp->getAttrs());
      rewriter.replaceOp(callOp, newOp);

      for (auto a : allocas) {
        if (a.first->hasOneUse()) {
          // If the alloca is only used for a store and the call operand, the
          // store is no longer required.
          rewriter.eraseOp(a.second);
          rewriter.eraseOp(a.first);
        }
      }
      LLVM_DEBUG(llvm::dbgs() << "global constant for " << callOp << " as "
                              << newOp << '\n');
      return mlir::success();
    }

    // Failure here just means "we couldn't do the conversion", which is
    // perfectly acceptable to the upper layers of this function.
    return mlir::failure();
  }
};

// this pass attempts to convert immediate scalar literals in function calls
// to global constants to allow transformations such as Dead Argument
// Elimination
class ConstantArgumentGlobalisationOpt
    : public fir::impl::ConstantArgumentGlobalisationOptBase<
          ConstantArgumentGlobalisationOpt> {
public:
  ConstantArgumentGlobalisationOpt() = default;

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::DominanceInfo *di = &getAnalysis<mlir::DominanceInfo>();
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    mlir::GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);
    config.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);

    patterns.insert<CallOpRewriter>(context, *di);
    if (mlir::failed(
            mlir::applyPatternsGreedily(mod, std::move(patterns), config))) {
      mlir::emitError(mod.getLoc(),
                      "error in constant globalisation optimization\n");
      signalPassFailure();
    }
  }
};
} // namespace
