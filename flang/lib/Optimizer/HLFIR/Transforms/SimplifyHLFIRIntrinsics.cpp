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

#define DEBUG_TYPE "simplify-hlfir-intrinsics"

static llvm::cl::opt<bool> forceMatmulAsElemental(
    "flang-inline-matmul-as-elemental",
    llvm::cl::desc("Expand hlfir.matmul as elemental operation"),
    llvm::cl::init(false));

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

template <typename Op>
class MatmulConversion : public mlir::OpRewritePattern<Op> {
public:
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(Op matmul, mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = matmul.getLoc();
    fir::FirOpBuilder builder{rewriter, matmul.getOperation()};
    hlfir::Entity lhs = hlfir::Entity{matmul.getLhs()};
    hlfir::Entity rhs = hlfir::Entity{matmul.getRhs()};
    mlir::Value resultShape, innerProductExtent;
    std::tie(resultShape, innerProductExtent) =
        genResultShape(loc, builder, lhs, rhs);

    if (forceMatmulAsElemental || isMatmulTranspose) {
      // Generate hlfir.elemental that produces the result of
      // MATMUL/MATMUL(TRANSPOSE).
      // Note that this implementation is very suboptimal for MATMUL,
      // but is quite good for MATMUL(TRANSPOSE), e.g.:
      //   R(1:N) = R(1:N) + MATMUL(TRANSPOSE(X(1:N,1:N)), Y(1:N))
      // Inlining MATMUL(TRANSPOSE) as hlfir.elemental may result
      // in merging the inner product computation with the elemental
      // addition. Note that the inner product computation will
      // benefit from processing the lowermost dimensions of X and Y,
      // which may be the best when they are contiguous.
      //
      // This is why we always inline MATMUL(TRANSPOSE) as an elemental.
      // MATMUL is inlined below by default unless forceMatmulAsElemental.
      hlfir::ExprType resultType =
          mlir::cast<hlfir::ExprType>(matmul.getType());
      hlfir::ElementalOp newOp = genElementalMatmul(
          loc, builder, resultType, resultShape, lhs, rhs, innerProductExtent);
      rewriter.replaceOp(matmul, newOp);
      return mlir::success();
    }

    // Generate hlfir.eval_in_mem to mimic the MATMUL implementation
    // from Fortran runtime. The implementation needs to operate
    // with the result array as an in-memory object.
    hlfir::EvaluateInMemoryOp evalOp =
        builder.create<hlfir::EvaluateInMemoryOp>(
            loc, mlir::cast<hlfir::ExprType>(matmul.getType()), resultShape);
    builder.setInsertionPointToStart(&evalOp.getBody().front());

    // Embox the raw array pointer to simplify designating it.
    // TODO: this currently results in redundant lower bounds
    // addition for the designator, but this should be fixed in
    // hlfir::Entity::mayHaveNonDefaultLowerBounds().
    mlir::Value resultArray = evalOp.getMemory();
    mlir::Type arrayType = fir::dyn_cast_ptrEleTy(resultArray.getType());
    resultArray = builder.createBox(loc, fir::BoxType::get(arrayType),
                                    resultArray, resultShape, /*slice=*/nullptr,
                                    /*lengths=*/{}, /*tdesc=*/nullptr);

    // The contiguous MATMUL version is best for the cases
    // where the input arrays and (maybe) the result are contiguous
    // in their lowermost dimensions.
    // Especially, when LLVM can recognize the continuity
    // and vectorize the loops properly.
    // Note that the contiguous MATMUL inlining is correct
    // even when the input arrays are not contiguous.
    // TODO: we can try to recognize the cases when the continuity
    // is not statically obvious and try to generate an explicitly
    // continuous version under a dynamic check. This should allow
    // LLVM to vectorize the loops better. Note that this can
    // also be postponed up to the LoopVersioning pass.
    // The fallback implementation may use genElementalMatmul() with
    // an hlfir.assign into the result of eval_in_mem.
    mlir::LogicalResult rewriteResult =
        genContiguousMatmul(loc, builder, hlfir::Entity{resultArray},
                            resultShape, lhs, rhs, innerProductExtent);

    if (mlir::failed(rewriteResult)) {
      // Erase the unclaimed eval_in_mem op.
      rewriter.eraseOp(evalOp);
      return rewriter.notifyMatchFailure(matmul,
                                         "genContiguousMatmul() failed");
    }

    rewriter.replaceOp(matmul, evalOp);
    return mlir::success();
  }

private:
  static constexpr bool isMatmulTranspose =
      std::is_same_v<Op, hlfir::MatmulTransposeOp>;

  // Return a tuple of:
  //   * A fir.shape operation representing the shape of the result
  //     of a MATMUL/MATMUL(TRANSPOSE).
  //   * An extent of the dimensions of the input array
  //     that are processed during the inner product computation.
  static std::tuple<mlir::Value, mlir::Value>
  genResultShape(mlir::Location loc, fir::FirOpBuilder &builder,
                 hlfir::Entity input1, hlfir::Entity input2) {
    mlir::Value input1Shape = hlfir::genShape(loc, builder, input1);
    llvm::SmallVector<mlir::Value> input1Extents =
        hlfir::getExplicitExtentsFromShape(input1Shape, builder);
    if (input1Shape.getUses().empty())
      input1Shape.getDefiningOp()->erase();
    mlir::Value input2Shape = hlfir::genShape(loc, builder, input2);
    llvm::SmallVector<mlir::Value> input2Extents =
        hlfir::getExplicitExtentsFromShape(input2Shape, builder);
    if (input2Shape.getUses().empty())
      input2Shape.getDefiningOp()->erase();

    llvm::SmallVector<mlir::Value, 2> newExtents;
    mlir::Value innerProduct1Extent, innerProduct2Extent;
    if (input1Extents.size() == 1) {
      assert(!isMatmulTranspose &&
             "hlfir.matmul_transpose's first operand must be rank-2 array");
      assert(input2Extents.size() == 2 &&
             "hlfir.matmul second argument must be rank-2 array");
      newExtents.push_back(input2Extents[1]);
      innerProduct1Extent = input1Extents[0];
      innerProduct2Extent = input2Extents[0];
    } else {
      if (input2Extents.size() == 1) {
        assert(input1Extents.size() == 2 &&
               "hlfir.matmul first argument must be rank-2 array");
        if constexpr (isMatmulTranspose)
          newExtents.push_back(input1Extents[1]);
        else
          newExtents.push_back(input1Extents[0]);
      } else {
        assert(input1Extents.size() == 2 && input2Extents.size() == 2 &&
               "hlfir.matmul arguments must be rank-2 arrays");
        if constexpr (isMatmulTranspose)
          newExtents.push_back(input1Extents[1]);
        else
          newExtents.push_back(input1Extents[0]);

        newExtents.push_back(input2Extents[1]);
      }
      if constexpr (isMatmulTranspose)
        innerProduct1Extent = input1Extents[0];
      else
        innerProduct1Extent = input1Extents[1];

      innerProduct2Extent = input2Extents[0];
    }
    // The inner product dimensions of the input arrays
    // must match. Pick the best (e.g. constant) out of them
    // so that the inner product loop bound can be used in
    // optimizations.
    llvm::SmallVector<mlir::Value> innerProductExtent =
        fir::factory::deduceOptimalExtents({innerProduct1Extent},
                                           {innerProduct2Extent});
    return {builder.create<fir::ShapeOp>(loc, newExtents),
            innerProductExtent[0]};
  }

  static mlir::Value castToProductType(mlir::Location loc,
                                       fir::FirOpBuilder &builder,
                                       mlir::Value value, mlir::Type type) {
    if (mlir::isa<fir::LogicalType>(type))
      return builder.createConvert(loc, builder.getIntegerType(1), value);

    // TODO: the multiplications/additions by/of zero resulting from
    // complex * real are optimized by LLVM under -fno-signed-zeros
    // -fno-honor-nans.
    // We can make them disappear by default if we:
    //   * either expand the complex multiplication into real
    //     operations, OR
    //   * set nnan nsz fast-math flags to the complex operations.
    if (fir::isa_complex(type) && !fir::isa_complex(value.getType())) {
      mlir::Value zeroCmplx = fir::factory::createZeroValue(builder, loc, type);
      fir::factory::Complex helper(builder, loc);
      mlir::Type partType = helper.getComplexPartType(type);
      return helper.insertComplexPart(
          zeroCmplx, castToProductType(loc, builder, value, partType),
          /*isImagPart=*/false);
    }
    return builder.createConvert(loc, type, value);
  }

  // Generate an update of the inner product value:
  //   acc += v1 * v2, OR
  //   acc ||= v1 && v2
  static mlir::Value genAccumulateProduct(mlir::Location loc,
                                          fir::FirOpBuilder &builder,
                                          mlir::Type resultType,
                                          mlir::Value acc, mlir::Value v1,
                                          mlir::Value v2) {
    acc = castToProductType(loc, builder, acc, resultType);
    v1 = castToProductType(loc, builder, v1, resultType);
    v2 = castToProductType(loc, builder, v2, resultType);
    mlir::Value result;
    if (mlir::isa<mlir::FloatType>(resultType))
      result = builder.create<mlir::arith::AddFOp>(
          loc, acc, builder.create<mlir::arith::MulFOp>(loc, v1, v2));
    else if (mlir::isa<mlir::ComplexType>(resultType))
      result = builder.create<fir::AddcOp>(
          loc, acc, builder.create<fir::MulcOp>(loc, v1, v2));
    else if (mlir::isa<mlir::IntegerType>(resultType))
      result = builder.create<mlir::arith::AddIOp>(
          loc, acc, builder.create<mlir::arith::MulIOp>(loc, v1, v2));
    else if (mlir::isa<fir::LogicalType>(resultType))
      result = builder.create<mlir::arith::OrIOp>(
          loc, acc, builder.create<mlir::arith::AndIOp>(loc, v1, v2));
    else
      llvm_unreachable("unsupported type");

    return builder.createConvert(loc, resultType, result);
  }

  static mlir::LogicalResult
  genContiguousMatmul(mlir::Location loc, fir::FirOpBuilder &builder,
                      hlfir::Entity result, mlir::Value resultShape,
                      hlfir::Entity lhs, hlfir::Entity rhs,
                      mlir::Value innerProductExtent) {
    // This code does not support MATMUL(TRANSPOSE), and it is supposed
    // to be inlined as hlfir.elemental.
    if constexpr (isMatmulTranspose)
      return mlir::failure();

    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Type resultElementType = result.getFortranElementType();
    llvm::SmallVector<mlir::Value, 2> resultExtents =
        mlir::cast<fir::ShapeOp>(resultShape.getDefiningOp()).getExtents();

    // The inner product loop may be unordered if FastMathFlags::reassoc
    // transformations are allowed. The integer/logical inner product is
    // always unordered.
    // Note that isUnordered is currently applied to all loops
    // in the loop nests generated below, while it has to be applied
    // only to one.
    bool isUnordered = mlir::isa<mlir::IntegerType>(resultElementType) ||
                       mlir::isa<fir::LogicalType>(resultElementType) ||
                       static_cast<bool>(builder.getFastMathFlags() &
                                         mlir::arith::FastMathFlags::reassoc);

    // Insert the initialization loop nest that fills the whole result with
    // zeroes.
    mlir::Value initValue =
        fir::factory::createZeroValue(builder, loc, resultElementType);
    auto genInitBody = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                           mlir::ValueRange oneBasedIndices,
                           mlir::ValueRange reductionArgs)
        -> llvm::SmallVector<mlir::Value, 0> {
      hlfir::Entity resultElement =
          hlfir::getElementAt(loc, builder, result, oneBasedIndices);
      builder.create<hlfir::AssignOp>(loc, initValue, resultElement);
      return {};
    };

    hlfir::genLoopNestWithReductions(loc, builder, resultExtents,
                                     /*reductionInits=*/{}, genInitBody,
                                     /*isUnordered=*/true);

    if (lhs.getRank() == 2 && rhs.getRank() == 2) {
      //   LHS(NROWS,N) * RHS(N,NCOLS) -> RESULT(NROWS,NCOLS)
      //
      // Insert the computation loop nest:
      //   DO 2 K = 1, N
      //    DO 2 J = 1, NCOLS
      //     DO 2 I = 1, NROWS
      //   2  RESULT(I,J) = RESULT(I,J) + LHS(I,K)*RHS(K,J)
      auto genMatrixMatrix = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                                 mlir::ValueRange oneBasedIndices,
                                 mlir::ValueRange reductionArgs)
          -> llvm::SmallVector<mlir::Value, 0> {
        mlir::Value I = oneBasedIndices[0];
        mlir::Value J = oneBasedIndices[1];
        mlir::Value K = oneBasedIndices[2];
        hlfir::Entity resultElement =
            hlfir::getElementAt(loc, builder, result, {I, J});
        hlfir::Entity resultElementValue =
            hlfir::loadTrivialScalar(loc, builder, resultElement);
        hlfir::Entity lhsElementValue =
            hlfir::loadElementAt(loc, builder, lhs, {I, K});
        hlfir::Entity rhsElementValue =
            hlfir::loadElementAt(loc, builder, rhs, {K, J});
        mlir::Value productValue = genAccumulateProduct(
            loc, builder, resultElementType, resultElementValue,
            lhsElementValue, rhsElementValue);
        builder.create<hlfir::AssignOp>(loc, productValue, resultElement);
        return {};
      };

      // Note that the loops are inserted in reverse order,
      // so innerProductExtent should be passed as the last extent.
      hlfir::genLoopNestWithReductions(
          loc, builder,
          {resultExtents[0], resultExtents[1], innerProductExtent},
          /*reductionInits=*/{}, genMatrixMatrix, isUnordered);
      return mlir::success();
    }

    if (lhs.getRank() == 2 && rhs.getRank() == 1) {
      //   LHS(NROWS,N) * RHS(N) -> RESULT(NROWS)
      //
      // Insert the computation loop nest:
      //   DO 2 K = 1, N
      //    DO 2 J = 1, NROWS
      //   2 RES(J) = RES(J) + LHS(J,K)*RHS(K)
      auto genMatrixVector = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                                 mlir::ValueRange oneBasedIndices,
                                 mlir::ValueRange reductionArgs)
          -> llvm::SmallVector<mlir::Value, 0> {
        mlir::Value J = oneBasedIndices[0];
        mlir::Value K = oneBasedIndices[1];
        hlfir::Entity resultElement =
            hlfir::getElementAt(loc, builder, result, {J});
        hlfir::Entity resultElementValue =
            hlfir::loadTrivialScalar(loc, builder, resultElement);
        hlfir::Entity lhsElementValue =
            hlfir::loadElementAt(loc, builder, lhs, {J, K});
        hlfir::Entity rhsElementValue =
            hlfir::loadElementAt(loc, builder, rhs, {K});
        mlir::Value productValue = genAccumulateProduct(
            loc, builder, resultElementType, resultElementValue,
            lhsElementValue, rhsElementValue);
        builder.create<hlfir::AssignOp>(loc, productValue, resultElement);
        return {};
      };
      hlfir::genLoopNestWithReductions(
          loc, builder, {resultExtents[0], innerProductExtent},
          /*reductionInits=*/{}, genMatrixVector, isUnordered);
      return mlir::success();
    }
    if (lhs.getRank() == 1 && rhs.getRank() == 2) {
      //   LHS(N) * RHS(N,NCOLS) -> RESULT(NCOLS)
      //
      // Insert the computation loop nest:
      //   DO 2 K = 1, N
      //    DO 2 J = 1, NCOLS
      //   2 RES(J) = RES(J) + LHS(K)*RHS(K,J)
      auto genVectorMatrix = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                                 mlir::ValueRange oneBasedIndices,
                                 mlir::ValueRange reductionArgs)
          -> llvm::SmallVector<mlir::Value, 0> {
        mlir::Value J = oneBasedIndices[0];
        mlir::Value K = oneBasedIndices[1];
        hlfir::Entity resultElement =
            hlfir::getElementAt(loc, builder, result, {J});
        hlfir::Entity resultElementValue =
            hlfir::loadTrivialScalar(loc, builder, resultElement);
        hlfir::Entity lhsElementValue =
            hlfir::loadElementAt(loc, builder, lhs, {K});
        hlfir::Entity rhsElementValue =
            hlfir::loadElementAt(loc, builder, rhs, {K, J});
        mlir::Value productValue = genAccumulateProduct(
            loc, builder, resultElementType, resultElementValue,
            lhsElementValue, rhsElementValue);
        builder.create<hlfir::AssignOp>(loc, productValue, resultElement);
        return {};
      };
      hlfir::genLoopNestWithReductions(
          loc, builder, {resultExtents[0], innerProductExtent},
          /*reductionInits=*/{}, genVectorMatrix, isUnordered);
      return mlir::success();
    }

    llvm_unreachable("unsupported MATMUL arguments' ranks");
  }

  static hlfir::ElementalOp
  genElementalMatmul(mlir::Location loc, fir::FirOpBuilder &builder,
                     hlfir::ExprType resultType, mlir::Value resultShape,
                     hlfir::Entity lhs, hlfir::Entity rhs,
                     mlir::Value innerProductExtent) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Type resultElementType = resultType.getElementType();
    auto genKernel = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                         mlir::ValueRange resultIndices) -> hlfir::Entity {
      mlir::Value initValue =
          fir::factory::createZeroValue(builder, loc, resultElementType);
      // The inner product loop may be unordered if FastMathFlags::reassoc
      // transformations are allowed. The integer/logical inner product is
      // always unordered.
      bool isUnordered = mlir::isa<mlir::IntegerType>(resultElementType) ||
                         mlir::isa<fir::LogicalType>(resultElementType) ||
                         static_cast<bool>(builder.getFastMathFlags() &
                                           mlir::arith::FastMathFlags::reassoc);

      auto genBody = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                         mlir::ValueRange oneBasedIndices,
                         mlir::ValueRange reductionArgs)
          -> llvm::SmallVector<mlir::Value, 1> {
        llvm::SmallVector<mlir::Value, 2> lhsIndices;
        llvm::SmallVector<mlir::Value, 2> rhsIndices;
        // MATMUL:
        //   LHS(NROWS,N) * RHS(N,NCOLS) -> RESULT(NROWS,NCOLS)
        //   LHS(NROWS,N) * RHS(N) -> RESULT(NROWS)
        //   LHS(N) * RHS(N,NCOLS) -> RESULT(NCOLS)
        //
        // MATMUL(TRANSPOSE):
        //   TRANSPOSE(LHS(N,NROWS)) * RHS(N,NCOLS) -> RESULT(NROWS,NCOLS)
        //   TRANSPOSE(LHS(N,NROWS)) * RHS(N) -> RESULT(NROWS)
        //
        // The resultIndices iterate over (NROWS[,NCOLS]).
        // The oneBasedIndices iterate over (N).
        if (lhs.getRank() > 1)
          lhsIndices.push_back(resultIndices[0]);
        lhsIndices.push_back(oneBasedIndices[0]);

        if constexpr (isMatmulTranspose) {
          // Swap the LHS indices for TRANSPOSE.
          std::swap(lhsIndices[0], lhsIndices[1]);
        }

        rhsIndices.push_back(oneBasedIndices[0]);
        if (rhs.getRank() > 1)
          rhsIndices.push_back(resultIndices.back());

        hlfir::Entity lhsElementValue =
            hlfir::loadElementAt(loc, builder, lhs, lhsIndices);
        hlfir::Entity rhsElementValue =
            hlfir::loadElementAt(loc, builder, rhs, rhsIndices);
        mlir::Value productValue = genAccumulateProduct(
            loc, builder, resultElementType, reductionArgs[0], lhsElementValue,
            rhsElementValue);
        return {productValue};
      };
      llvm::SmallVector<mlir::Value, 1> innerProductValue =
          hlfir::genLoopNestWithReductions(loc, builder, {innerProductExtent},
                                           {initValue}, genBody, isUnordered);
      return hlfir::Entity{innerProductValue[0]};
    };
    hlfir::ElementalOp elementalOp = hlfir::genElementalOp(
        loc, builder, resultElementType, resultShape, /*typeParams=*/{},
        genKernel,
        /*isUnordered=*/true, /*polymorphicMold=*/nullptr, resultType);

    return elementalOp;
  }
};

class SimplifyHLFIRIntrinsics
    : public hlfir::impl::SimplifyHLFIRIntrinsicsBase<SimplifyHLFIRIntrinsics> {
public:
  using SimplifyHLFIRIntrinsicsBase<
      SimplifyHLFIRIntrinsics>::SimplifyHLFIRIntrinsicsBase;

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
    patterns.insert<MatmulConversion<hlfir::MatmulTransposeOp>>(context);

    // If forceMatmulAsElemental is false, then hlfir.matmul inlining
    // will introduce hlfir.eval_in_mem operation with new memory side
    // effects. This conflicts with CSE and optimized bufferization, e.g.:
    //   A(1:N,1:N) =  A(1:N,1:N) - MATMUL(...)
    // If we introduce hlfir.eval_in_mem before CSE, then the current
    // MLIR CSE won't be able to optimize the trivial loads of 'N' value
    // that happen before and after hlfir.matmul.
    // If 'N' loads are not optimized, then the optimized bufferization
    // won't be able to prove that the slices of A are identical
    // on both sides of the assignment.
    // This is actually the CSE problem, but we can work it around
    // for the time being.
    if (forceMatmulAsElemental || this->allowNewSideEffects)
      patterns.insert<MatmulConversion<hlfir::MatmulOp>>(context);

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in HLFIR intrinsic simplification");
      signalPassFailure();
    }
  }
};
} // namespace
