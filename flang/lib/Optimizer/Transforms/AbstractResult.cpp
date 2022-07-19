//===- AbstractResult.cpp - Conversion of Abstract Function Result --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "flang-abstract-result-opt"

namespace fir {
namespace {

static mlir::Type getResultArgumentType(mlir::Type resultType,
                                        bool shouldBoxResult) {
  return llvm::TypeSwitch<mlir::Type, mlir::Type>(resultType)
      .Case<fir::SequenceType, fir::RecordType>(
          [&](mlir::Type type) -> mlir::Type {
            if (shouldBoxResult)
              return fir::BoxType::get(type);
            return fir::ReferenceType::get(type);
          })
      .Case<fir::BoxType>([](mlir::Type type) -> mlir::Type {
        return fir::ReferenceType::get(type);
      })
      .Default([](mlir::Type) -> mlir::Type {
        llvm_unreachable("bad abstract result type");
      });
}

static mlir::FunctionType getNewFunctionType(mlir::FunctionType funcTy,
                                             bool shouldBoxResult) {
  auto resultType = funcTy.getResult(0);
  auto argTy = getResultArgumentType(resultType, shouldBoxResult);
  llvm::SmallVector<mlir::Type> newInputTypes = {argTy};
  newInputTypes.append(funcTy.getInputs().begin(), funcTy.getInputs().end());
  return mlir::FunctionType::get(funcTy.getContext(), newInputTypes,
                                 /*resultTypes=*/{});
}

static bool mustEmboxResult(mlir::Type resultType, bool shouldBoxResult) {
  return resultType.isa<fir::SequenceType, fir::RecordType>() &&
         shouldBoxResult;
}

class CallOpConversion : public mlir::OpRewritePattern<fir::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  CallOpConversion(mlir::MLIRContext *context, bool shouldBoxResult)
      : OpRewritePattern(context), shouldBoxResult{shouldBoxResult} {}
  mlir::LogicalResult
  matchAndRewrite(fir::CallOp callOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = callOp.getLoc();
    auto result = callOp->getResult(0);
    if (!result.hasOneUse()) {
      mlir::emitError(loc,
                      "calls with abstract result must have exactly one user");
      return mlir::failure();
    }
    auto saveResult =
        mlir::dyn_cast<fir::SaveResultOp>(result.use_begin().getUser());
    if (!saveResult) {
      mlir::emitError(
          loc, "calls with abstract result must be used in fir.save_result");
      return mlir::failure();
    }
    auto argType = getResultArgumentType(result.getType(), shouldBoxResult);
    auto buffer = saveResult.getMemref();
    mlir::Value arg = buffer;
    if (mustEmboxResult(result.getType(), shouldBoxResult))
      arg = rewriter.create<fir::EmboxOp>(
          loc, argType, buffer, saveResult.getShape(), /*slice*/ mlir::Value{},
          saveResult.getTypeparams());

    llvm::SmallVector<mlir::Type> newResultTypes;
    if (callOp.getCallee()) {
      llvm::SmallVector<mlir::Value> newOperands = {arg};
      newOperands.append(callOp.getOperands().begin(),
                         callOp.getOperands().end());
      rewriter.create<fir::CallOp>(loc, *callOp.getCallee(), newResultTypes,
                                   newOperands);
    } else {
      // Indirect calls.
      llvm::SmallVector<mlir::Type> newInputTypes = {argType};
      for (auto operand : callOp.getOperands().drop_front())
        newInputTypes.push_back(operand.getType());
      auto funTy = mlir::FunctionType::get(callOp.getContext(), newInputTypes,
                                           newResultTypes);

      llvm::SmallVector<mlir::Value> newOperands;
      newOperands.push_back(
          rewriter.create<fir::ConvertOp>(loc, funTy, callOp.getOperand(0)));
      newOperands.push_back(arg);
      newOperands.append(callOp.getOperands().begin() + 1,
                         callOp.getOperands().end());
      rewriter.create<fir::CallOp>(loc, mlir::SymbolRefAttr{}, newResultTypes,
                                   newOperands);
    }
    callOp->dropAllReferences();
    rewriter.eraseOp(callOp);
    return mlir::success();
  }

private:
  bool shouldBoxResult;
};

class SaveResultOpConversion
    : public mlir::OpRewritePattern<fir::SaveResultOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  SaveResultOpConversion(mlir::MLIRContext *context)
      : OpRewritePattern(context) {}
  mlir::LogicalResult
  matchAndRewrite(fir::SaveResultOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ReturnOpConversion : public mlir::OpRewritePattern<mlir::func::ReturnOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  ReturnOpConversion(mlir::MLIRContext *context, mlir::Value newArg)
      : OpRewritePattern(context), newArg{newArg} {}
  mlir::LogicalResult
  matchAndRewrite(mlir::func::ReturnOp ret,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(ret);
    auto returnedValue = ret.getOperand(0);
    bool replacedStorage = false;
    if (auto *op = returnedValue.getDefiningOp())
      if (auto load = mlir::dyn_cast<fir::LoadOp>(op)) {
        auto resultStorage = load.getMemref();
        load.getMemref().replaceAllUsesWith(newArg);
        replacedStorage = true;
        if (auto *alloc = resultStorage.getDefiningOp())
          if (alloc->use_empty())
            rewriter.eraseOp(alloc);
      }
    // The result storage may have been optimized out by a memory to
    // register pass, this is possible for fir.box results, or fir.record
    // with no length parameters. Simply store the result in the result storage.
    // at the return point.
    if (!replacedStorage)
      rewriter.create<fir::StoreOp>(ret.getLoc(), returnedValue, newArg);
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(ret);
    return mlir::success();
  }

private:
  mlir::Value newArg;
};

class AddrOfOpConversion : public mlir::OpRewritePattern<fir::AddrOfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  AddrOfOpConversion(mlir::MLIRContext *context, bool shouldBoxResult)
      : OpRewritePattern(context), shouldBoxResult{shouldBoxResult} {}
  mlir::LogicalResult
  matchAndRewrite(fir::AddrOfOp addrOf,
                  mlir::PatternRewriter &rewriter) const override {
    auto oldFuncTy = addrOf.getType().cast<mlir::FunctionType>();
    auto newFuncTy = getNewFunctionType(oldFuncTy, shouldBoxResult);
    auto newAddrOf = rewriter.create<fir::AddrOfOp>(addrOf.getLoc(), newFuncTy,
                                                    addrOf.getSymbol());
    // Rather than converting all op a function pointer might transit through
    // (e.g calls, stores, loads, converts...), cast new type to the abstract
    // type. A conversion will be added when calling indirect calls of abstract
    // types.
    rewriter.replaceOpWithNewOp<fir::ConvertOp>(addrOf, oldFuncTy, newAddrOf);
    return mlir::success();
  }

private:
  bool shouldBoxResult;
};

class AbstractResultOpt : public fir::AbstractResultOptBase<AbstractResultOpt> {
public:
  void runOnOperation() override {
    auto *context = &getContext();
    auto func = getOperation();
    auto loc = func.getLoc();
    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target = *context;
    const bool shouldBoxResult = passResultAsBox.getValue();

    // Convert function type itself if it has an abstract result
    auto funcTy = func.getFunctionType().cast<mlir::FunctionType>();
    if (hasAbstractResult(funcTy)) {
      func.setType(getNewFunctionType(funcTy, shouldBoxResult));
      unsigned zero = 0;
      if (!func.empty()) {
        // Insert new argument
        mlir::OpBuilder rewriter(context);
        auto resultType = funcTy.getResult(0);
        auto argTy = getResultArgumentType(resultType, shouldBoxResult);
        mlir::Value newArg = func.front().insertArgument(zero, argTy, loc);
        if (mustEmboxResult(resultType, shouldBoxResult)) {
          auto bufferType = fir::ReferenceType::get(resultType);
          rewriter.setInsertionPointToStart(&func.front());
          newArg = rewriter.create<fir::BoxAddrOp>(loc, bufferType, newArg);
        }
        patterns.insert<ReturnOpConversion>(context, newArg);
        target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
            [](mlir::func::ReturnOp ret) { return ret.operands().empty(); });
      }
    }

    if (func.empty())
      return;

    // Convert the calls and, if needed,  the ReturnOp in the function body.
    target.addLegalDialect<fir::FIROpsDialect, mlir::arith::ArithmeticDialect,
                           mlir::func::FuncDialect>();
    target.addIllegalOp<fir::SaveResultOp>();
    target.addDynamicallyLegalOp<fir::CallOp>([](fir::CallOp call) {
      return !hasAbstractResult(call.getFunctionType());
    });
    target.addDynamicallyLegalOp<fir::AddrOfOp>([](fir::AddrOfOp addrOf) {
      if (auto funTy = addrOf.getType().dyn_cast<mlir::FunctionType>())
        return !hasAbstractResult(funTy);
      return true;
    });
    target.addDynamicallyLegalOp<fir::DispatchOp>([](fir::DispatchOp dispatch) {
      if (dispatch->getNumResults() != 1)
        return true;
      auto resultType = dispatch->getResult(0).getType();
      if (resultType.isa<fir::SequenceType, fir::BoxType, fir::RecordType>()) {
        TODO(dispatch.getLoc(), "dispatchOp with abstract results");
        return false;
      }
      return true;
    });

    patterns.insert<CallOpConversion>(context, shouldBoxResult);
    patterns.insert<SaveResultOpConversion>(context);
    patterns.insert<AddrOfOpConversion>(context, shouldBoxResult);
    if (mlir::failed(
            mlir::applyPartialConversion(func, target, std::move(patterns)))) {
      mlir::emitError(func.getLoc(), "error in converting abstract results\n");
      signalPassFailure();
    }
  }
};
} // end anonymous namespace
} // namespace fir

std::unique_ptr<mlir::Pass> fir::createAbstractResultOptPass() {
  return std::make_unique<AbstractResultOpt>();
}
