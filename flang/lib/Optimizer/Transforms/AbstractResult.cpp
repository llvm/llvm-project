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
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/LazySymbolTable.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace fir {
#define GEN_PASS_DEF_ABSTRACTRESULTOPT
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-abstract-result-opt"

using namespace aiir;

namespace fir {
namespace {

bool hasScalarDerivedResult(aiir::FunctionType funTy) {
  // C_PTR/C_FUNPTR are results to void* in this pass, do not consider
  // them as normal derived types.
  return funTy.getNumResults() == 1 &&
         aiir::isa<fir::RecordType>(funTy.getResult(0)) &&
         !fir::isa_builtin_cptr_type(funTy.getResult(0));
}

static aiir::Type getResultArgumentType(aiir::Type resultType,
                                        bool shouldBoxResult) {
  return llvm::TypeSwitch<aiir::Type, aiir::Type>(resultType)
      .Case<fir::SequenceType, fir::RecordType>(
          [&](aiir::Type type) -> aiir::Type {
            if (shouldBoxResult)
              return fir::BoxType::get(type);
            return fir::ReferenceType::get(type);
          })
      .Case<fir::BaseBoxType>([](aiir::Type type) -> aiir::Type {
        return fir::ReferenceType::get(type);
      })
      .Default([](aiir::Type) -> aiir::Type {
        llvm_unreachable("bad abstract result type");
      });
}

static aiir::FunctionType getNewFunctionType(aiir::FunctionType funcTy,
                                             bool shouldBoxResult) {
  auto resultType = funcTy.getResult(0);
  auto argTy = getResultArgumentType(resultType, shouldBoxResult);
  llvm::SmallVector<aiir::Type> newInputTypes = {argTy};
  newInputTypes.append(funcTy.getInputs().begin(), funcTy.getInputs().end());
  return aiir::FunctionType::get(funcTy.getContext(), newInputTypes,
                                 /*resultTypes=*/{});
}

static aiir::Type getVoidPtrType(aiir::AIIRContext *context) {
  return fir::ReferenceType::get(aiir::NoneType::get(context));
}

/// This is for function result types that are of type C_PTR from ISO_C_BINDING.
/// Follow the ABI for interoperability with C.
static aiir::FunctionType getCPtrFunctionType(aiir::FunctionType funcTy) {
  assert(fir::isa_builtin_cptr_type(funcTy.getResult(0)));
  llvm::SmallVector<aiir::Type> outputTypes{
      getVoidPtrType(funcTy.getContext())};
  return aiir::FunctionType::get(funcTy.getContext(), funcTy.getInputs(),
                                 outputTypes);
}

static bool mustEmboxResult(aiir::Type resultType, bool shouldBoxResult) {
  return aiir::isa<fir::SequenceType, fir::RecordType>(resultType) &&
         shouldBoxResult;
}

template <typename Op>
class CallConversion : public aiir::OpRewritePattern<Op> {
public:
  using aiir::OpRewritePattern<Op>::OpRewritePattern;

  CallConversion(aiir::AIIRContext *context, bool shouldBoxResult)
      : OpRewritePattern<Op>(context, 1), shouldBoxResult{shouldBoxResult} {}

  llvm::LogicalResult
  matchAndRewrite(Op op, aiir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto result = op->getResult(0);
    if (!result.hasOneUse()) {
      aiir::emitError(loc,
                      "calls with abstract result must have exactly one user");
      return aiir::failure();
    }
    auto saveResult =
        aiir::dyn_cast<fir::SaveResultOp>(result.use_begin().getUser());
    if (!saveResult) {
      aiir::emitError(
          loc, "calls with abstract result must be used in fir.save_result");
      return aiir::failure();
    }
    auto argType = getResultArgumentType(result.getType(), shouldBoxResult);
    auto buffer = saveResult.getMemref();
    aiir::Value arg = buffer;
    if (mustEmboxResult(result.getType(), shouldBoxResult))
      arg = fir::EmboxOp::create(rewriter, loc, argType, buffer,
                                 saveResult.getShape(), /*slice*/ aiir::Value{},
                                 saveResult.getTypeparams());

    llvm::SmallVector<aiir::Type> newResultTypes;
    bool isResultBuiltinCPtr = fir::isa_builtin_cptr_type(result.getType());
    if (isResultBuiltinCPtr)
      newResultTypes.emplace_back(getVoidPtrType(result.getContext()));

    Op newOp;
    // TODO: propagate argument and result attributes (need to be shifted).
    // fir::CallOp specific handling.
    if constexpr (std::is_same_v<Op, fir::CallOp>) {
      if (op.getCallee()) {
        llvm::SmallVector<aiir::Value> newOperands;
        if (!isResultBuiltinCPtr)
          newOperands.emplace_back(arg);
        newOperands.append(op.getOperands().begin(), op.getOperands().end());
        newOp = fir::CallOp::create(rewriter, loc, *op.getCallee(),
                                    newResultTypes, newOperands);
      } else {
        // Indirect calls.
        llvm::SmallVector<aiir::Type> newInputTypes;
        if (!isResultBuiltinCPtr)
          newInputTypes.emplace_back(argType);
        for (auto operand : op.getOperands().drop_front())
          newInputTypes.push_back(operand.getType());
        auto newFuncTy = aiir::FunctionType::get(op.getContext(), newInputTypes,
                                                 newResultTypes);

        llvm::SmallVector<aiir::Value> newOperands;
        newOperands.push_back(
            fir::ConvertOp::create(rewriter, loc, newFuncTy, op.getOperand(0)));
        if (!isResultBuiltinCPtr)
          newOperands.push_back(arg);
        newOperands.append(op.getOperands().begin() + 1,
                           op.getOperands().end());
        newOp = fir::CallOp::create(rewriter, loc, aiir::SymbolRefAttr{},
                                    newResultTypes, newOperands);
      }
    }

    // fir::DispatchOp specific handling.
    if constexpr (std::is_same_v<Op, fir::DispatchOp>) {
      llvm::SmallVector<aiir::Value> newOperands;
      if (!isResultBuiltinCPtr)
        newOperands.emplace_back(arg);
      unsigned passArgShift = newOperands.size();
      newOperands.append(op.getOperands().begin() + 1, op.getOperands().end());
      aiir::IntegerAttr passArgPos;
      if (op.getPassArgPos())
        passArgPos =
            rewriter.getI32IntegerAttr(*op.getPassArgPos() + passArgShift);
      // TODO: propagate argument and result attributes (need to be shifted).
      newOp = fir::DispatchOp::create(
          rewriter, loc, newResultTypes, rewriter.getStringAttr(op.getMethod()),
          op.getOperands()[0], newOperands, passArgPos,
          /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
          op.getProcedureAttrsAttr());
    }

    if (isResultBuiltinCPtr) {
      aiir::Value save = saveResult.getMemref();
      auto module = op->template getParentOfType<aiir::ModuleOp>();
      FirOpBuilder builder(rewriter, module);
      aiir::Value saveAddr = fir::factory::genCPtrOrCFunptrAddr(
          builder, loc, save, result.getType());
      builder.createStoreWithConvert(loc, newOp->getResult(0), saveAddr);
    }
    op->dropAllReferences();
    rewriter.eraseOp(op);
    return aiir::success();
  }

private:
  bool shouldBoxResult;
};

class SaveResultOpConversion
    : public aiir::OpRewritePattern<fir::SaveResultOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  SaveResultOpConversion(aiir::AIIRContext *context)
      : OpRewritePattern(context) {}
  llvm::LogicalResult
  matchAndRewrite(fir::SaveResultOp op,
                  aiir::PatternRewriter &rewriter) const override {
    aiir::Operation *call = op.getValue().getDefiningOp();
    aiir::Type type = op.getValue().getType();
    if (aiir::isa<fir::RecordType>(type) && call && fir::hasBindcAttr(call) &&
        !fir::isa_builtin_cptr_type(type)) {
      rewriter.replaceOpWithNewOp<fir::StoreOp>(op, op.getValue(),
                                                op.getMemref());
    } else {
      rewriter.eraseOp(op);
    }
    return aiir::success();
  }
};

template <typename OpTy>
static aiir::LogicalResult
processReturnLikeOp(OpTy ret, aiir::Value newArg,
                    aiir::PatternRewriter &rewriter) {
  auto loc = ret.getLoc();
  rewriter.setInsertionPoint(ret);
  aiir::Value resultValue = ret.getOperand(0);
  fir::LoadOp resultLoad;
  aiir::Value resultStorage;
  // Identify result local storage.
  if (auto load = resultValue.getDefiningOp<fir::LoadOp>()) {
    resultLoad = load;
    resultStorage = load.getMemref();
    // The result alloca may be behind a fir.declare, if any.
    if (auto declare = resultStorage.getDefiningOp<fir::DeclareOp>())
      resultStorage = declare.getMemref();
  }
  // Replace old local storage with new storage argument, unless
  // the derived type is C_PTR/C_FUN_PTR, in which case the return
  // type is updated to return void* (no new argument is passed).
  if (fir::isa_builtin_cptr_type(resultValue.getType())) {
    auto module = ret->template getParentOfType<aiir::ModuleOp>();
    FirOpBuilder builder(rewriter, module);
    aiir::Value cptr = resultValue;
    if (resultLoad) {
      // Replace whole derived type load by component load.
      cptr = resultLoad.getMemref();
      rewriter.setInsertionPoint(resultLoad);
    }
    aiir::Value newResultValue =
        fir::factory::genCPtrOrCFunptrValue(builder, loc, cptr);
    newResultValue = builder.createConvert(
        loc, getVoidPtrType(ret.getContext()), newResultValue);
    rewriter.setInsertionPoint(ret);
    rewriter.replaceOpWithNewOp<OpTy>(ret, aiir::ValueRange{newResultValue});
  } else if (resultStorage) {
    resultStorage.replaceAllUsesWith(newArg);
    rewriter.replaceOpWithNewOp<OpTy>(ret);
  } else {
    // The result storage may have been optimized out by a memory to
    // register pass, this is possible for fir.box results, or fir.record
    // with no length parameters. Simply store the result in the result
    // storage. at the return point.
    fir::StoreOp::create(rewriter, loc, resultValue, newArg);
    rewriter.replaceOpWithNewOp<OpTy>(ret);
  }
  // Delete result old local storage if unused.
  if (resultStorage)
    if (auto alloc = resultStorage.getDefiningOp<fir::AllocaOp>())
      if (alloc->use_empty())
        rewriter.eraseOp(alloc);
  return aiir::success();
}

class ReturnOpConversion : public aiir::OpRewritePattern<aiir::func::ReturnOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  ReturnOpConversion(aiir::AIIRContext *context, aiir::Value newArg)
      : OpRewritePattern(context), newArg{newArg} {}
  llvm::LogicalResult
  matchAndRewrite(aiir::func::ReturnOp ret,
                  aiir::PatternRewriter &rewriter) const override {
    return processReturnLikeOp(ret, newArg, rewriter);
  }

private:
  aiir::Value newArg;
};

class GPUReturnOpConversion
    : public aiir::OpRewritePattern<aiir::gpu::ReturnOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  GPUReturnOpConversion(aiir::AIIRContext *context, aiir::Value newArg)
      : OpRewritePattern(context), newArg{newArg} {}
  llvm::LogicalResult
  matchAndRewrite(aiir::gpu::ReturnOp ret,
                  aiir::PatternRewriter &rewriter) const override {
    return processReturnLikeOp(ret, newArg, rewriter);
  }

private:
  aiir::Value newArg;
};

class AddrOfOpConversion : public aiir::OpRewritePattern<fir::AddrOfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  AddrOfOpConversion(aiir::AIIRContext *context, bool shouldBoxResult)
      : OpRewritePattern(context), shouldBoxResult{shouldBoxResult} {}
  llvm::LogicalResult
  matchAndRewrite(fir::AddrOfOp addrOf,
                  aiir::PatternRewriter &rewriter) const override {
    auto oldFuncTy = aiir::cast<aiir::FunctionType>(addrOf.getType());
    aiir::FunctionType newFuncTy;
    if (oldFuncTy.getNumResults() != 0 &&
        fir::isa_builtin_cptr_type(oldFuncTy.getResult(0)))
      newFuncTy = getCPtrFunctionType(oldFuncTy);
    else
      newFuncTy = getNewFunctionType(oldFuncTy, shouldBoxResult);
    auto newAddrOf = fir::AddrOfOp::create(rewriter, addrOf.getLoc(), newFuncTy,
                                           addrOf.getSymbol());
    // Rather than converting all op a function pointer might transit through
    // (e.g calls, stores, loads, converts...), cast new type to the abstract
    // type. A conversion will be added when calling indirect calls of abstract
    // types.
    rewriter.replaceOpWithNewOp<fir::ConvertOp>(addrOf, oldFuncTy, newAddrOf);
    return aiir::success();
  }

private:
  bool shouldBoxResult;
};

class AbstractResultOpt
    : public fir::impl::AbstractResultOptBase<AbstractResultOpt> {
public:
  using fir::impl::AbstractResultOptBase<
      AbstractResultOpt>::AbstractResultOptBase;

  template <typename OpTy>
  void runOnFunctionLikeOperation(OpTy func, bool shouldBoxResult,
                                  aiir::RewritePatternSet &patterns,
                                  aiir::ConversionTarget &target) {
    auto loc = func.getLoc();
    auto *context = &getContext();
    // Convert function type itself if it has an abstract result.
    auto funcTy = aiir::cast<aiir::FunctionType>(func.getFunctionType());
    // Scalar derived result of BIND(C) function must be returned according
    // to the C struct return ABI which is target dependent and implemented in
    // the target-rewrite pass.
    if (hasScalarDerivedResult(funcTy) &&
        fir::hasBindcAttr(func.getOperation()))
      return;
    if (hasAbstractResult(funcTy)) {
      if (fir::isa_builtin_cptr_type(funcTy.getResult(0))) {
        func.setType(getCPtrFunctionType(funcTy));
        patterns.insert<ReturnOpConversion>(context, aiir::Value{});
        target.addDynamicallyLegalOp<aiir::func::ReturnOp>(
            [](aiir::func::ReturnOp ret) {
              aiir::Type retTy = ret.getOperand(0).getType();
              return !fir::isa_builtin_cptr_type(retTy);
            });
        return;
      }
      if (!func.empty()) {
        // Insert new argument.
        aiir::OpBuilder rewriter(context);
        auto resultType = funcTy.getResult(0);
        auto argTy = getResultArgumentType(resultType, shouldBoxResult);
        llvm::LogicalResult res = func.insertArgument(0u, argTy, {}, loc);
        (void)res;
        assert(llvm::succeeded(res) && "failed to insert function argument");
        res = func.eraseResult(0u);
        (void)res;
        assert(llvm::succeeded(res) && "failed to erase function result");
        aiir::Value newArg = func.getArgument(0u);
        if (mustEmboxResult(resultType, shouldBoxResult)) {
          auto bufferType = fir::ReferenceType::get(resultType);
          rewriter.setInsertionPointToStart(&func.front());
          newArg = fir::BoxAddrOp::create(rewriter, loc, bufferType, newArg);
        }
        patterns.insert<ReturnOpConversion>(context, newArg);
        target.addDynamicallyLegalOp<aiir::func::ReturnOp>(
            [](aiir::func::ReturnOp ret) { return ret.getOperands().empty(); });
        patterns.insert<GPUReturnOpConversion>(context, newArg);
        target.addDynamicallyLegalOp<aiir::gpu::ReturnOp>(
            [](aiir::gpu::ReturnOp ret) { return ret.getOperands().empty(); });
        assert(func.getFunctionType() ==
               getNewFunctionType(funcTy, shouldBoxResult));
      } else {
        llvm::SmallVector<aiir::DictionaryAttr> allArgs;
        func.getAllArgAttrs(allArgs);
        allArgs.insert(allArgs.begin(),
                       aiir::DictionaryAttr::get(func->getContext()));
        func.setType(getNewFunctionType(funcTy, shouldBoxResult));
        func.setAllArgAttrs(allArgs);
      }
    }
  }

  void runOnSpecificOperation(aiir::func::FuncOp func, bool shouldBoxResult,
                              aiir::RewritePatternSet &patterns,
                              aiir::ConversionTarget &target) {
    runOnFunctionLikeOperation(func, shouldBoxResult, patterns, target);
  }

  void runOnSpecificOperation(aiir::gpu::GPUFuncOp func, bool shouldBoxResult,
                              aiir::RewritePatternSet &patterns,
                              aiir::ConversionTarget &target) {
    runOnFunctionLikeOperation(func, shouldBoxResult, patterns, target);
  }

  inline static bool containsFunctionTypeWithAbstractResult(aiir::Type type) {
    return aiir::TypeSwitch<aiir::Type, bool>(type)
        .Case([](fir::BoxProcType boxProc) {
          return fir::hasAbstractResult(
              aiir::cast<aiir::FunctionType>(boxProc.getEleTy()));
        })
        .Case([](fir::PointerType pointer) {
          return fir::hasAbstractResult(
              aiir::cast<aiir::FunctionType>(pointer.getEleTy()));
        })
        .Default([](auto &&) { return false; });
  }

  void runOnSpecificOperation(fir::GlobalOp global, bool,
                              aiir::RewritePatternSet &,
                              aiir::ConversionTarget &) {
    if (containsFunctionTypeWithAbstractResult(global.getType())) {
      TODO(global->getLoc(), "support for procedure pointers");
    }
  }

  /// Run the pass on a ModuleOp. This makes fir-opt --abstract-result work.
  void runOnModule() {
    aiir::ModuleOp mod = aiir::cast<aiir::ModuleOp>(getOperation());

    auto pass = std::make_unique<AbstractResultOpt>();
    pass->copyOptionValuesFrom(this);
    aiir::OpPassManager pipeline;
    pipeline.addPass(std::unique_ptr<aiir::Pass>{pass.release()});

    // Run the pass on all operations directly nested inside of the ModuleOp
    // we can't just call runOnSpecificOperation here because the pass
    // implementation only works when scoped to a particular func.func or
    // fir.global
    for (aiir::Region &region : mod->getRegions()) {
      for (aiir::Block &block : region.getBlocks()) {
        for (aiir::Operation &op : block.getOperations()) {
          if (aiir::failed(runPipeline(pipeline, &op))) {
            aiir::emitError(op.getLoc(), "Failed to run abstract result pass");
            signalPassFailure();
            return;
          }
        }
      }
    }
  }

  void runOnOperation() override {
    auto *context = &this->getContext();
    aiir::Operation *op = this->getOperation();
    if (aiir::isa<aiir::ModuleOp>(op)) {
      runOnModule();
      return;
    }

    fir::LazySymbolTable symbolTable(op);

    aiir::RewritePatternSet patterns(context);
    aiir::ConversionTarget target = *context;
    const bool shouldBoxResult = this->passResultAsBox.getValue();

    aiir::TypeSwitch<aiir::Operation *, void>(op)
        .Case<aiir::func::FuncOp, fir::GlobalOp, aiir::gpu::GPUFuncOp>(
            [&](auto op) {
              runOnSpecificOperation(op, shouldBoxResult, patterns, target);
            });

    // Convert the calls and, if needed,  the ReturnOp in the function body.
    target.addLegalDialect<fir::FIROpsDialect, aiir::arith::ArithDialect,
                           aiir::func::FuncDialect>();
    target.addIllegalOp<fir::SaveResultOp>();
    target.addDynamicallyLegalOp<fir::CallOp>([](fir::CallOp call) {
      aiir::FunctionType funTy = call.getFunctionType();
      if (hasScalarDerivedResult(funTy) &&
          fir::hasBindcAttr(call.getOperation()))
        return true;
      return !hasAbstractResult(funTy);
    });
    target.addDynamicallyLegalOp<fir::AddrOfOp>([&symbolTable](
                                                    fir::AddrOfOp addrOf) {
      if (auto funTy = aiir::dyn_cast<aiir::FunctionType>(addrOf.getType())) {
        if (hasScalarDerivedResult(funTy)) {
          auto func = symbolTable.lookup<aiir::func::FuncOp>(
              addrOf.getSymbol().getRootReference().getValue());
          return func && fir::hasBindcAttr(func.getOperation());
        }
        return !hasAbstractResult(funTy);
      }
      return true;
    });
    target.addDynamicallyLegalOp<fir::DispatchOp>([](fir::DispatchOp dispatch) {
      aiir::FunctionType funTy = dispatch.getFunctionType();
      if (hasScalarDerivedResult(funTy) &&
          fir::hasBindcAttr(dispatch.getOperation()))
        return true;
      return !hasAbstractResult(dispatch.getFunctionType());
    });

    patterns.insert<CallConversion<fir::CallOp>>(context, shouldBoxResult);
    patterns.insert<CallConversion<fir::DispatchOp>>(context, shouldBoxResult);
    patterns.insert<SaveResultOpConversion>(context);
    patterns.insert<AddrOfOpConversion>(context, shouldBoxResult);
    if (aiir::failed(
            aiir::applyPartialConversion(op, target, std::move(patterns)))) {
      aiir::emitError(op->getLoc(), "error in converting abstract results\n");
      this->signalPassFailure();
    }
  }
};

} // end anonymous namespace
} // namespace fir
