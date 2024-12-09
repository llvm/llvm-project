//===- SimplifyHLFIRIntrinsics.cpp - Simplify HLFIR Intrinsics ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Normally transformational intrinsics are lowered to calls to runtime
// functions. However, some cases of the intrinsics are faster when inlined
// into the calling function.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace hlfir {
#define GEN_PASS_DEF_SIMPLIFYHLFIRINTRINSICS
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

namespace {

class TransposeAsElementalConversion
    : public mlir::OpRewritePattern<hlfir::TransposeOp> {
public:
  using mlir::OpRewritePattern<hlfir::TransposeOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::TransposeOp transpose,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = transpose.getLoc();
    fir::FirOpBuilder builder{rewriter, transpose.getOperation()};
    hlfir::ExprType expr = transpose.getType();
    mlir::Type elementType = expr.getElementType();
    hlfir::Entity array = hlfir::Entity{transpose.getArray()};
    mlir::Value resultShape = genResultShape(loc, builder, array);
    llvm::SmallVector<mlir::Value, 1> typeParams;
    hlfir::genLengthParameters(loc, builder, array, typeParams);

    auto genKernel = [&array](mlir::Location loc, fir::FirOpBuilder &builder,
                              mlir::ValueRange inputIndices) -> hlfir::Entity {
      assert(inputIndices.size() == 2 && "checked in TransposeOp::validate");
      const std::initializer_list<mlir::Value> initList = {inputIndices[1],
                                                           inputIndices[0]};
      mlir::ValueRange transposedIndices(initList);
      hlfir::Entity element =
          hlfir::getElementAt(loc, builder, array, transposedIndices);
      hlfir::Entity val = hlfir::loadTrivialScalar(loc, builder, element);
      return val;
    };
    hlfir::ElementalOp elementalOp = hlfir::genElementalOp(
        loc, builder, elementType, resultShape, typeParams, genKernel,
        /*isUnordered=*/true, /*polymorphicMold=*/nullptr,
        transpose.getResult().getType());

    // it wouldn't be safe to replace block arguments with a different
    // hlfir.expr type. Types can differ due to differing amounts of shape
    // information
    assert(elementalOp.getResult().getType() ==
           transpose.getResult().getType());

    rewriter.replaceOp(transpose, elementalOp);
    return mlir::success();
  }

private:
  static mlir::Value genResultShape(mlir::Location loc,
                                    fir::FirOpBuilder &builder,
                                    hlfir::Entity array) {
    mlir::Value inShape = hlfir::genShape(loc, builder, array);
    llvm::SmallVector<mlir::Value> inExtents =
        hlfir::getExplicitExtentsFromShape(inShape, builder);
    if (inShape.getUses().empty())
      inShape.getDefiningOp()->erase();

    // transpose indices
    assert(inExtents.size() == 2 && "checked in TransposeOp::validate");
    return builder.create<fir::ShapeOp>(
        loc, mlir::ValueRange{inExtents[1], inExtents[0]});
  }
};

// Expand the SUM(DIM=CONSTANT) operation into .
class SumAsElementalConversion : public mlir::OpRewritePattern<hlfir::SumOp> {
public:
  using mlir::OpRewritePattern<hlfir::SumOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::SumOp sum,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = sum.getLoc();
    fir::FirOpBuilder builder{rewriter, sum.getOperation()};
    hlfir::ExprType expr = mlir::dyn_cast<hlfir::ExprType>(sum.getType());
    assert(expr && "expected an expression type for the result of hlfir.sum");
    mlir::Type elementType = expr.getElementType();
    hlfir::Entity array = hlfir::Entity{sum.getArray()};
    mlir::Value mask = sum.getMask();
    mlir::Value dim = sum.getDim();
    int64_t dimVal = fir::getIntIfConstant(dim).value_or(0);
    mlir::Value resultShape, dimExtent;
    std::tie(resultShape, dimExtent) =
        genResultShape(loc, builder, array, dimVal);

    auto genKernel = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                         mlir::ValueRange inputIndices) -> hlfir::Entity {
      // Loop over all indices in the DIM dimension, and reduce all values.
      // We do not need to create the reduction loop always: if we can
      // slice the input array given the inputIndices, then we can
      // just apply a new SUM operation (total reduction) to the slice.
      // For the time being, generate the explicit loop because the slicing
      // requires generating an elemental operation for the input array
      // (and the mask, if present).
      // TODO: produce the slices and new SUM after adding a pattern
      // for expanding total reduction SUM case.
      mlir::Type indexType = builder.getIndexType();
      auto one = builder.createIntegerConstant(loc, indexType, 1);
      auto ub = builder.createConvert(loc, indexType, dimExtent);

      // Initial value for the reduction.
      mlir::Value initValue = genInitValue(loc, builder, elementType);

      // The reduction loop may be unordered if FastMathFlags::reassoc
      // transformations are allowed. The integer reduction is always
      // unordered.
      bool isUnordered = mlir::isa<mlir::IntegerType>(elementType) ||
                         static_cast<bool>(sum.getFastmath() &
                                           mlir::arith::FastMathFlags::reassoc);

      // If the mask is present and is a scalar, then we'd better load its value
      // outside of the reduction loop making the loop unswitching easier.
      // Maybe it is worth hoisting it from the elemental operation as well.
      mlir::Value isPresentPred, maskValue;
      if (mask) {
        if (mlir::isa<fir::BaseBoxType>(mask.getType())) {
          // MASK represented by a box might be dynamically optional,
          // so we have to check for its presence before accessing it.
          isPresentPred =
              builder.create<fir::IsPresentOp>(loc, builder.getI1Type(), mask);
        }

        if (hlfir::Entity{mask}.isScalar())
          maskValue = genMaskValue(loc, builder, mask, isPresentPred, {});
      }

      // NOTE: the outer elemental operation may be lowered into
      // omp.workshare.loop_wrapper/omp.loop_nest later, so the reduction
      // loop may appear disjoint from the workshare loop nest.
      // Moreover, the inner loop is not strictly nested (due to the reduction
      // starting value initialization), and the above omp dialect operations
      // cannot produce results.
      // It is unclear what we should do about it yet.
      auto doLoop = builder.create<fir::DoLoopOp>(
          loc, one, ub, one, isUnordered, /*finalCountValue=*/false,
          mlir::ValueRange{initValue});

      // Address the input array using the reduction loop's IV
      // for the DIM dimension.
      mlir::Value iv = doLoop.getInductionVar();
      llvm::SmallVector<mlir::Value> indices{inputIndices};
      indices.insert(indices.begin() + dimVal - 1, iv);

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(doLoop.getBody());
      mlir::Value reductionValue = doLoop.getRegionIterArgs()[0];
      fir::IfOp ifOp;
      if (mask) {
        // Make the reduction value update conditional on the value
        // of the mask.
        if (!maskValue) {
          // If the mask is an array, use the elemental and the loop indices
          // to address the proper mask element.
          maskValue = genMaskValue(loc, builder, mask, isPresentPred, indices);
        }
        mlir::Value isUnmasked =
            builder.create<fir::ConvertOp>(loc, builder.getI1Type(), maskValue);
        ifOp = builder.create<fir::IfOp>(loc, elementType, isUnmasked,
                                         /*withElseRegion=*/true);
        // In the 'else' block return the current reduction value.
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        builder.create<fir::ResultOp>(loc, reductionValue);

        // In the 'then' block do the actual addition.
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      }

      hlfir::Entity element = hlfir::getElementAt(loc, builder, array, indices);
      hlfir::Entity elementValue =
          hlfir::loadTrivialScalar(loc, builder, element);
      // NOTE: we can use "Kahan summation" same way as the runtime
      // (e.g. when fast-math is not allowed), but let's start with
      // the simple version.
      reductionValue = genScalarAdd(loc, builder, reductionValue, elementValue);
      builder.create<fir::ResultOp>(loc, reductionValue);

      if (ifOp) {
        builder.setInsertionPointAfter(ifOp);
        builder.create<fir::ResultOp>(loc, ifOp.getResult(0));
      }

      return hlfir::Entity{doLoop.getResult(0)};
    };
    hlfir::ElementalOp elementalOp = hlfir::genElementalOp(
        loc, builder, elementType, resultShape, {}, genKernel,
        /*isUnordered=*/true, /*polymorphicMold=*/nullptr,
        sum.getResult().getType());

    // it wouldn't be safe to replace block arguments with a different
    // hlfir.expr type. Types can differ due to differing amounts of shape
    // information
    assert(elementalOp.getResult().getType() == sum.getResult().getType());

    rewriter.replaceOp(sum, elementalOp);
    return mlir::success();
  }

private:
  // Return fir.shape specifying the shape of the result
  // of a SUM reduction with DIM=dimVal. The second return value
  // is the extent of the DIM dimension.
  static std::tuple<mlir::Value, mlir::Value>
  genResultShape(mlir::Location loc, fir::FirOpBuilder &builder,
                 hlfir::Entity array, int64_t dimVal) {
    mlir::Value inShape = hlfir::genShape(loc, builder, array);
    llvm::SmallVector<mlir::Value> inExtents =
        hlfir::getExplicitExtentsFromShape(inShape, builder);
    assert(dimVal > 0 && dimVal <= static_cast<int64_t>(inExtents.size()) &&
           "DIM must be present and a positive constant not exceeding "
           "the array's rank");
    if (inShape.getUses().empty())
      inShape.getDefiningOp()->erase();

    mlir::Value dimExtent = inExtents[dimVal - 1];
    inExtents.erase(inExtents.begin() + dimVal - 1);
    return {builder.create<fir::ShapeOp>(loc, inExtents), dimExtent};
  }

  // Generate the initial value for a SUM reduction with the given
  // data type.
  static mlir::Value genInitValue(mlir::Location loc,
                                  fir::FirOpBuilder &builder,
                                  mlir::Type elementType) {
    if (auto ty = mlir::dyn_cast<mlir::FloatType>(elementType)) {
      const llvm::fltSemantics &sem = ty.getFloatSemantics();
      return builder.createRealConstant(loc, elementType,
                                        llvm::APFloat::getZero(sem));
    } else if (auto ty = mlir::dyn_cast<mlir::ComplexType>(elementType)) {
      mlir::Value initValue = genInitValue(loc, builder, ty.getElementType());
      return fir::factory::Complex{builder, loc}.createComplex(ty, initValue,
                                                               initValue);
    } else if (mlir::isa<mlir::IntegerType>(elementType)) {
      return builder.createIntegerConstant(loc, elementType, 0);
    }

    llvm_unreachable("unsupported SUM reduction type");
  }

  // Generate scalar addition of the two values (of the same data type).
  static mlir::Value genScalarAdd(mlir::Location loc,
                                  fir::FirOpBuilder &builder,
                                  mlir::Value value1, mlir::Value value2) {
    mlir::Type ty = value1.getType();
    assert(ty == value2.getType() && "reduction values' types do not match");
    if (mlir::isa<mlir::FloatType>(ty))
      return builder.create<mlir::arith::AddFOp>(loc, value1, value2);
    else if (mlir::isa<mlir::ComplexType>(ty))
      return builder.create<fir::AddcOp>(loc, value1, value2);
    else if (mlir::isa<mlir::IntegerType>(ty))
      return builder.create<mlir::arith::AddIOp>(loc, value1, value2);

    llvm_unreachable("unsupported SUM reduction type");
  }

  static mlir::Value genMaskValue(mlir::Location loc,
                                  fir::FirOpBuilder &builder, mlir::Value mask,
                                  mlir::Value isPresentPred,
                                  mlir::ValueRange indices) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    fir::IfOp ifOp;
    mlir::Type maskType =
        hlfir::getFortranElementType(fir::unwrapPassByRefType(mask.getType()));
    if (isPresentPred) {
      ifOp = builder.create<fir::IfOp>(loc, maskType, isPresentPred,
                                       /*withElseRegion=*/true);

      // Use 'true', if the mask is not present.
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      mlir::Value trueValue = builder.createBool(loc, true);
      trueValue = builder.createConvert(loc, maskType, trueValue);
      builder.create<fir::ResultOp>(loc, trueValue);

      // Load the mask value, if the mask is present.
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    }

    hlfir::Entity maskVar{mask};
    if (maskVar.isScalar()) {
      if (mlir::isa<fir::BaseBoxType>(mask.getType())) {
        // MASK may be a boxed scalar.
        mlir::Value addr = hlfir::genVariableRawAddress(loc, builder, maskVar);
        mask = builder.create<fir::LoadOp>(loc, hlfir::Entity{addr});
      } else {
        mask = hlfir::loadTrivialScalar(loc, builder, maskVar);
      }
    } else {
      // Load from the mask array.
      assert(!indices.empty() && "no indices for addressing the mask array");
      maskVar = hlfir::getElementAt(loc, builder, maskVar, indices);
      mask = hlfir::loadTrivialScalar(loc, builder, maskVar);
    }

    if (!isPresentPred)
      return mask;

    builder.create<fir::ResultOp>(loc, mask);
    return ifOp.getResult(0);
  }
};

class SimplifyHLFIRIntrinsics
    : public hlfir::impl::SimplifyHLFIRIntrinsicsBase<SimplifyHLFIRIntrinsics> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<TransposeAsElementalConversion>(context);
    patterns.insert<SumAsElementalConversion>(context);
    mlir::ConversionTarget target(*context);
    // don't transform transpose of polymorphic arrays (not currently supported
    // by hlfir.elemental)
    target.addDynamicallyLegalOp<hlfir::TransposeOp>(
        [](hlfir::TransposeOp transpose) {
          return mlir::cast<hlfir::ExprType>(transpose.getType())
              .isPolymorphic();
        });
    // Handle only SUM(DIM=CONSTANT) case for now.
    // It may be beneficial to expand the non-DIM case as well.
    // E.g. when the input array is an elemental array expression,
    // expanding the SUM into a total reduction loop nest
    // would avoid creating a temporary for the elemental array expression.
    target.addDynamicallyLegalOp<hlfir::SumOp>([](hlfir::SumOp sum) {
      if (mlir::Value dim = sum.getDim()) {
        if (auto dimVal = fir::getIntIfConstant(dim)) {
          if (!fir::isa_trivial(sum.getType())) {
            // Ignore the case SUM(a, DIM=X), where 'a' is a 1D array.
            // It is only legal when X is 1, and it should probably be
            // canonicalized into SUM(a).
            fir::SequenceType arrayTy = mlir::cast<fir::SequenceType>(
                hlfir::getFortranElementOrSequenceType(
                    sum.getArray().getType()));
            if (*dimVal > 0 && *dimVal <= arrayTy.getDimension()) {
              // Ignore SUMs with illegal DIM values.
              // They may appear in dead code,
              // and they do not have to be converted.
              return false;
            }
          }
        }
      }
      return true;
    });
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });
    if (mlir::failed(mlir::applyFullConversion(getOperation(), target,
                                               std::move(patterns)))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in HLFIR intrinsic simplification");
      signalPassFailure();
    }
  }
};
} // namespace
