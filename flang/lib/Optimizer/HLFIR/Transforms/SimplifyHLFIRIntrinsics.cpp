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
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
    hlfir::ExprType expr = transpose.getType();
    // TODO: hlfir.elemental supports polymorphic data types now,
    // so this can be supported.
    if (expr.isPolymorphic())
      return rewriter.notifyMatchFailure(transpose,
                                         "TRANSPOSE of polymorphic type");

    mlir::Location loc = transpose.getLoc();
    fir::FirOpBuilder builder{rewriter, transpose.getOperation()};
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
    hlfir::Entity array = hlfir::Entity{sum.getArray()};
    bool isTotalReduction = hlfir::Entity{sum}.getRank() == 0;
    mlir::Value dim = sum.getDim();
    int64_t dimVal = 0;
    if (!isTotalReduction) {
      // In case of partial reduction we should ignore the operations
      // with invalid DIM values. They may appear in dead code
      // after constant propagation.
      auto constDim = fir::getIntIfConstant(dim);
      if (!constDim)
        return rewriter.notifyMatchFailure(sum, "Nonconstant DIM for SUM");
      dimVal = *constDim;

      if ((dimVal <= 0 || dimVal > array.getRank()))
        return rewriter.notifyMatchFailure(
            sum, "Invalid DIM for partial SUM reduction");
    }

    mlir::Location loc = sum.getLoc();
    fir::FirOpBuilder builder{rewriter, sum.getOperation()};
    mlir::Type elementType = hlfir::getFortranElementType(sum.getType());
    mlir::Value mask = sum.getMask();

    mlir::Value resultShape, dimExtent;
    llvm::SmallVector<mlir::Value> arrayExtents;
    if (isTotalReduction)
      arrayExtents = genArrayExtents(loc, builder, array);
    else
      std::tie(resultShape, dimExtent) =
          genResultShapeForPartialReduction(loc, builder, array, dimVal);

    // If the mask is present and is a scalar, then we'd better load its value
    // outside of the reduction loop making the loop unswitching easier.
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

    auto genKernel = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                         mlir::ValueRange inputIndices) -> hlfir::Entity {
      // Loop over all indices in the DIM dimension, and reduce all values.
      // If DIM is not present, do total reduction.

      // Initial value for the reduction.
      mlir::Value reductionInitValue = genInitValue(loc, builder, elementType);

      // The reduction loop may be unordered if FastMathFlags::reassoc
      // transformations are allowed. The integer reduction is always
      // unordered.
      bool isUnordered = mlir::isa<mlir::IntegerType>(elementType) ||
                         static_cast<bool>(sum.getFastmath() &
                                           mlir::arith::FastMathFlags::reassoc);

      llvm::SmallVector<mlir::Value> extents;
      if (isTotalReduction)
        extents = arrayExtents;
      else
        extents.push_back(
            builder.createConvert(loc, builder.getIndexType(), dimExtent));

      auto genBody = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                         mlir::ValueRange oneBasedIndices,
                         mlir::ValueRange reductionArgs)
          -> llvm::SmallVector<mlir::Value, 1> {
        // Generate the reduction loop-nest body.
        // The initial reduction value in the innermost loop
        // is passed via reductionArgs[0].
        llvm::SmallVector<mlir::Value> indices;
        if (isTotalReduction) {
          indices = oneBasedIndices;
        } else {
          indices = inputIndices;
          indices.insert(indices.begin() + dimVal - 1, oneBasedIndices[0]);
        }

        mlir::Value reductionValue = reductionArgs[0];
        fir::IfOp ifOp;
        if (mask) {
          // Make the reduction value update conditional on the value
          // of the mask.
          if (!maskValue) {
            // If the mask is an array, use the elemental and the loop indices
            // to address the proper mask element.
            maskValue =
                genMaskValue(loc, builder, mask, isPresentPred, indices);
          }
          mlir::Value isUnmasked = builder.create<fir::ConvertOp>(
              loc, builder.getI1Type(), maskValue);
          ifOp = builder.create<fir::IfOp>(loc, elementType, isUnmasked,
                                           /*withElseRegion=*/true);
          // In the 'else' block return the current reduction value.
          builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
          builder.create<fir::ResultOp>(loc, reductionValue);

          // In the 'then' block do the actual addition.
          builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        }

        hlfir::Entity element =
            hlfir::getElementAt(loc, builder, array, indices);
        hlfir::Entity elementValue =
            hlfir::loadTrivialScalar(loc, builder, element);
        // NOTE: we can use "Kahan summation" same way as the runtime
        // (e.g. when fast-math is not allowed), but let's start with
        // the simple version.
        reductionValue =
            genScalarAdd(loc, builder, reductionValue, elementValue);

        if (ifOp) {
          builder.create<fir::ResultOp>(loc, reductionValue);
          builder.setInsertionPointAfter(ifOp);
          reductionValue = ifOp.getResult(0);
        }

        return {reductionValue};
      };

      llvm::SmallVector<mlir::Value, 1> reductionFinalValues =
          hlfir::genLoopNestWithReductions(loc, builder, extents,
                                           {reductionInitValue}, genBody,
                                           isUnordered);
      return hlfir::Entity{reductionFinalValues[0]};
    };

    if (isTotalReduction) {
      hlfir::Entity result = genKernel(loc, builder, mlir::ValueRange{});
      rewriter.replaceOp(sum, result);
      return mlir::success();
    }

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
  static llvm::SmallVector<mlir::Value>
  genArrayExtents(mlir::Location loc, fir::FirOpBuilder &builder,
                  hlfir::Entity array) {
    mlir::Value inShape = hlfir::genShape(loc, builder, array);
    llvm::SmallVector<mlir::Value> inExtents =
        hlfir::getExplicitExtentsFromShape(inShape, builder);
    if (inShape.getUses().empty())
      inShape.getDefiningOp()->erase();
    return inExtents;
  }

  // Return fir.shape specifying the shape of the result
  // of a SUM reduction with DIM=dimVal. The second return value
  // is the extent of the DIM dimension.
  static std::tuple<mlir::Value, mlir::Value>
  genResultShapeForPartialReduction(mlir::Location loc,
                                    fir::FirOpBuilder &builder,
                                    hlfir::Entity array, int64_t dimVal) {
    llvm::SmallVector<mlir::Value> inExtents =
        genArrayExtents(loc, builder, array);
    assert(dimVal > 0 && dimVal <= static_cast<int64_t>(inExtents.size()) &&
           "DIM must be present and a positive constant not exceeding "
           "the array's rank");

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

class CShiftAsElementalConversion
    : public mlir::OpRewritePattern<hlfir::CShiftOp> {
public:
  using mlir::OpRewritePattern<hlfir::CShiftOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::CShiftOp cshift,
                  mlir::PatternRewriter &rewriter) const override {
    using Fortran::common::maxRank;

    hlfir::ExprType expr = mlir::dyn_cast<hlfir::ExprType>(cshift.getType());
    assert(expr &&
           "expected an expression type for the result of hlfir.cshift");
    unsigned arrayRank = expr.getRank();
    // When it is a 1D CSHIFT, we may assume that the DIM argument
    // (whether it is present or absent) is equal to 1, otherwise,
    // the program is illegal.
    int64_t dimVal = 1;
    if (arrayRank != 1)
      if (mlir::Value dim = cshift.getDim()) {
        auto constDim = fir::getIntIfConstant(dim);
        if (!constDim)
          return rewriter.notifyMatchFailure(cshift,
                                             "Nonconstant DIM for CSHIFT");
        dimVal = *constDim;
      }

    if (dimVal <= 0 || dimVal > arrayRank)
      return rewriter.notifyMatchFailure(cshift, "Invalid DIM for CSHIFT");

    mlir::Location loc = cshift.getLoc();
    fir::FirOpBuilder builder{rewriter, cshift.getOperation()};
    mlir::Type elementType = expr.getElementType();
    hlfir::Entity array = hlfir::Entity{cshift.getArray()};
    mlir::Value arrayShape = hlfir::genShape(loc, builder, array);
    llvm::SmallVector<mlir::Value> arrayExtents =
        hlfir::getExplicitExtentsFromShape(arrayShape, builder);
    llvm::SmallVector<mlir::Value, 1> typeParams;
    hlfir::genLengthParameters(loc, builder, array, typeParams);
    hlfir::Entity shift = hlfir::Entity{cshift.getShift()};
    // The new index computation involves MODULO, which is not implemented
    // for IndexType, so use I64 instead.
    mlir::Type calcType = builder.getI64Type();

    mlir::Value one = builder.createIntegerConstant(loc, calcType, 1);
    mlir::Value shiftVal;
    if (shift.isScalar()) {
      shiftVal = hlfir::loadTrivialScalar(loc, builder, shift);
      shiftVal = builder.createConvert(loc, calcType, shiftVal);
    }

    auto genKernel = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                         mlir::ValueRange inputIndices) -> hlfir::Entity {
      llvm::SmallVector<mlir::Value, maxRank> indices{inputIndices};
      if (!shift.isScalar()) {
        // When the array is not a vector, section
        // (s(1), s(2), ..., s(dim-1), :, s(dim+1), ..., s(n)
        // of the result has a value equal to:
        // CSHIFT(ARRAY(s(1), s(2), ..., s(dim-1), :, s(dim+1), ..., s(n)),
        //        SH, 1),
        // where SH is either SHIFT (if scalar) or
        // SHIFT(s(1), s(2), ..., s(dim-1), s(dim+1), ..., s(n)).
        llvm::SmallVector<mlir::Value, maxRank> shiftIndices{indices};
        shiftIndices.erase(shiftIndices.begin() + dimVal - 1);
        hlfir::Entity shiftElement =
            hlfir::getElementAt(loc, builder, shift, shiftIndices);
        shiftVal = hlfir::loadTrivialScalar(loc, builder, shiftElement);
        shiftVal = builder.createConvert(loc, calcType, shiftVal);
      }

      // Element i of the result (1-based) is element
      // 'MODULO(i + SH - 1, SIZE(ARRAY)) + 1' (1-based) of the original
      // ARRAY (or its section, when ARRAY is not a vector).
      mlir::Value index =
          builder.createConvert(loc, calcType, inputIndices[dimVal - 1]);
      mlir::Value extent = arrayExtents[dimVal - 1];
      mlir::Value newIndex =
          builder.create<mlir::arith::AddIOp>(loc, index, shiftVal);
      newIndex = builder.create<mlir::arith::SubIOp>(loc, newIndex, one);
      newIndex = fir::IntrinsicLibrary{builder, loc}.genModulo(
          calcType, {newIndex, builder.createConvert(loc, calcType, extent)});
      newIndex = builder.create<mlir::arith::AddIOp>(loc, newIndex, one);
      newIndex = builder.createConvert(loc, builder.getIndexType(), newIndex);

      indices[dimVal - 1] = newIndex;
      hlfir::Entity element = hlfir::getElementAt(loc, builder, array, indices);
      return hlfir::loadTrivialScalar(loc, builder, element);
    };

    hlfir::ElementalOp elementalOp = hlfir::genElementalOp(
        loc, builder, elementType, arrayShape, typeParams, genKernel,
        /*isUnordered=*/true,
        array.isPolymorphic() ? static_cast<mlir::Value>(array) : nullptr,
        cshift.getResult().getType());
    rewriter.replaceOp(cshift, elementalOp);
    return mlir::success();
  }
};

class SimplifyHLFIRIntrinsics
    : public hlfir::impl::SimplifyHLFIRIntrinsicsBase<SimplifyHLFIRIntrinsics> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();

    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks
    config.enableRegionSimplification =
        mlir::GreedySimplifyRegionLevel::Disabled;

    mlir::RewritePatternSet patterns(context);
    patterns.insert<TransposeAsElementalConversion>(context);
    patterns.insert<SumAsElementalConversion>(context);
    patterns.insert<CShiftAsElementalConversion>(context);

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in HLFIR intrinsic simplification");
      signalPassFailure();
    }
  }
};
} // namespace
