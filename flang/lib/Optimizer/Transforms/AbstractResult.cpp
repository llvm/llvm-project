//===- AbstractResult.cpp - Conversion of Abstract Function Result --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

namespace fir {
#define GEN_PASS_DEF_ABSTRACTRESULTONFUNCOPT
#define GEN_PASS_DEF_ABSTRACTRESULTONGLOBALOPT
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

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

/// This is for function result types that are of type C_PTR from ISO_C_BINDING.
/// Follow the ABI for interoperability with C.
static mlir::FunctionType getCPtrFunctionType(mlir::FunctionType funcTy) {
  auto resultType = funcTy.getResult(0);
  assert(fir::isa_builtin_cptr_type(resultType));
  llvm::SmallVector<mlir::Type> outputTypes;
  auto recTy = resultType.dyn_cast<fir::RecordType>();
  outputTypes.emplace_back(recTy.getTypeList()[0].second);
  return mlir::FunctionType::get(funcTy.getContext(), funcTy.getInputs(),
                                 outputTypes);
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
    // TODO: This should be generalized for derived types, and it is
    // architecture and OS dependent.
    bool isResultBuiltinCPtr = fir::isa_builtin_cptr_type(result.getType());
    fir::CallOp newCallOp;
    if (isResultBuiltinCPtr) {
      auto recTy = result.getType().dyn_cast<fir::RecordType>();
      newResultTypes.emplace_back(recTy.getTypeList()[0].second);
    }
    if (callOp.getCallee()) {
      llvm::SmallVector<mlir::Value> newOperands;
      if (!isResultBuiltinCPtr)
        newOperands.emplace_back(arg);
      newOperands.append(callOp.getOperands().begin(),
                         callOp.getOperands().end());
      newCallOp = rewriter.create<fir::CallOp>(loc, *callOp.getCallee(),
                                               newResultTypes, newOperands);
    } else {
      // Indirect calls.
      llvm::SmallVector<mlir::Type> newInputTypes;
      if (!isResultBuiltinCPtr)
        newInputTypes.emplace_back(argType);
      for (auto operand : callOp.getOperands().drop_front())
        newInputTypes.push_back(operand.getType());
      auto newFuncTy = mlir::FunctionType::get(callOp.getContext(),
                                               newInputTypes, newResultTypes);

      llvm::SmallVector<mlir::Value> newOperands;
      newOperands.push_back(rewriter.create<fir::ConvertOp>(
          loc, newFuncTy, callOp.getOperand(0)));
      if (!isResultBuiltinCPtr)
        newOperands.push_back(arg);
      newOperands.append(callOp.getOperands().begin() + 1,
                         callOp.getOperands().end());
      newCallOp = rewriter.create<fir::CallOp>(loc, mlir::SymbolRefAttr{},
                                               newResultTypes, newOperands);
    }
    if (isResultBuiltinCPtr) {
      mlir::Value save = saveResult.getMemref();
      auto module = callOp->getParentOfType<mlir::ModuleOp>();
      fir::KindMapping kindMap = fir::getKindMapping(module);
      FirOpBuilder builder(rewriter, kindMap);
      mlir::Value saveAddr = fir::factory::genCPtrOrCFunptrAddr(
          builder, loc, save, result.getType());
      rewriter.create<fir::StoreOp>(loc, newCallOp->getResult(0), saveAddr);
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
    auto loc = ret.getLoc();
    rewriter.setInsertionPoint(ret);
    auto returnedValue = ret.getOperand(0);
    bool replacedStorage = false;
    if (auto *op = returnedValue.getDefiningOp())
      if (auto load = mlir::dyn_cast<fir::LoadOp>(op)) {
        auto resultStorage = load.getMemref();
        // TODO: This should be generalized for derived types, and it is
        // architecture and OS dependent.
        if (fir::isa_builtin_cptr_type(returnedValue.getType())) {
          rewriter.eraseOp(load);
          auto module = ret->getParentOfType<mlir::ModuleOp>();
          fir::KindMapping kindMap = fir::getKindMapping(module);
          FirOpBuilder builder(rewriter, kindMap);
          mlir::Value retAddr = fir::factory::genCPtrOrCFunptrAddr(
              builder, loc, resultStorage, returnedValue.getType());
          mlir::Value retValue = rewriter.create<fir::LoadOp>(
              loc, fir::unwrapRefType(retAddr.getType()), retAddr);
          rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
              ret, mlir::ValueRange{retValue});
          return mlir::success();
        }
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
      rewriter.create<fir::StoreOp>(loc, returnedValue, newArg);
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
    mlir::FunctionType newFuncTy;
    // TODO: This should be generalized for derived types, and it is
    // architecture and OS dependent.
    if (oldFuncTy.getNumResults() != 0 &&
        fir::isa_builtin_cptr_type(oldFuncTy.getResult(0)))
      newFuncTy = getCPtrFunctionType(oldFuncTy);
    else
      newFuncTy = getNewFunctionType(oldFuncTy, shouldBoxResult);
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

/// @brief Base CRTP class for AbstractResult pass family.
/// Contains common logic for abstract result conversion in a reusable fashion.
/// @tparam Pass target class that implements operation-specific logic.
/// @tparam PassBase base class template for the pass generated by TableGen.
/// The `Pass` class must define runOnSpecificOperation(OpTy, bool,
/// mlir::RewritePatternSet&, mlir::ConversionTarget&) member function.
/// This function should implement operation-specific functionality.
template <typename Pass, template <typename> class PassBase>
class AbstractResultOptTemplate : public PassBase<Pass> {
public:
  void runOnOperation() override {
    auto *context = &this->getContext();
    auto op = this->getOperation();

    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target = *context;
    const bool shouldBoxResult = this->passResultAsBox.getValue();

    auto &self = static_cast<Pass &>(*this);
    self.runOnSpecificOperation(op, shouldBoxResult, patterns, target);

    // Convert the calls and, if needed,  the ReturnOp in the function body.
    target.addLegalDialect<fir::FIROpsDialect, mlir::arith::ArithDialect,
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
            mlir::applyPartialConversion(op, target, std::move(patterns)))) {
      mlir::emitError(op.getLoc(), "error in converting abstract results\n");
      this->signalPassFailure();
    }
  }
};

class AbstractResultOnFuncOpt
    : public AbstractResultOptTemplate<AbstractResultOnFuncOpt,
                                       fir::impl::AbstractResultOnFuncOptBase> {
public:
  void runOnSpecificOperation(mlir::func::FuncOp func, bool shouldBoxResult,
                              mlir::RewritePatternSet &patterns,
                              mlir::ConversionTarget &target) {
    auto loc = func.getLoc();
    auto *context = &getContext();
    // Convert function type itself if it has an abstract result.
    auto funcTy = func.getFunctionType().cast<mlir::FunctionType>();
    if (hasAbstractResult(funcTy)) {
      // TODO: This should be generalized for derived types, and it is
      // architecture and OS dependent.
      if (fir::isa_builtin_cptr_type(funcTy.getResult(0))) {
        func.setType(getCPtrFunctionType(funcTy));
        patterns.insert<ReturnOpConversion>(context, mlir::Value{});
        target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
            [](mlir::func::ReturnOp ret) {
              mlir::Type retTy = ret.getOperand(0).getType();
              return !fir::isa_builtin_cptr_type(retTy);
            });
        return;
      }
      if (!func.empty()) {
        // Insert new argument.
        mlir::OpBuilder rewriter(context);
        auto resultType = funcTy.getResult(0);
        auto argTy = getResultArgumentType(resultType, shouldBoxResult);
        func.insertArgument(0u, argTy, {}, loc);
        func.eraseResult(0u);
        mlir::Value newArg = func.getArgument(0u);
        if (mustEmboxResult(resultType, shouldBoxResult)) {
          auto bufferType = fir::ReferenceType::get(resultType);
          rewriter.setInsertionPointToStart(&func.front());
          newArg = rewriter.create<fir::BoxAddrOp>(loc, bufferType, newArg);
        }
        patterns.insert<ReturnOpConversion>(context, newArg);
        target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
            [](mlir::func::ReturnOp ret) { return ret.operands().empty(); });
        assert(func.getFunctionType() ==
               getNewFunctionType(funcTy, shouldBoxResult));
      } else {
        func.setType(getNewFunctionType(funcTy, shouldBoxResult));
      }
    }
  }
};

inline static bool containsFunctionTypeWithAbstractResult(mlir::Type type) {
  return mlir::TypeSwitch<mlir::Type, bool>(type)
      .Case([](fir::BoxProcType boxProc) {
        return fir::hasAbstractResult(
            boxProc.getEleTy().cast<mlir::FunctionType>());
      })
      .Case([](fir::PointerType pointer) {
        return fir::hasAbstractResult(
            pointer.getEleTy().cast<mlir::FunctionType>());
      })
      .Default([](auto &&) { return false; });
}

class AbstractResultOnGlobalOpt
    : public AbstractResultOptTemplate<
          AbstractResultOnGlobalOpt, fir::impl::AbstractResultOnGlobalOptBase> {
public:
  void runOnSpecificOperation(fir::GlobalOp global, bool,
                              mlir::RewritePatternSet &,
                              mlir::ConversionTarget &) {
    if (containsFunctionTypeWithAbstractResult(global.getType())) {
      TODO(global->getLoc(), "support for procedure pointers");
    }
  }
};
} // end anonymous namespace
} // namespace fir

std::unique_ptr<mlir::Pass> fir::createAbstractResultOnFuncOptPass() {
  return std::make_unique<AbstractResultOnFuncOpt>();
}

std::unique_ptr<mlir::Pass> fir::createAbstractResultOnGlobalOptPass() {
  return std::make_unique<AbstractResultOnGlobalOpt>();
}
