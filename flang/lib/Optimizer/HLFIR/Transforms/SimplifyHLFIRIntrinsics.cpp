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
#include "flang/Optimizer/OpenMP/Passes.h"
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

static llvm::cl::opt<bool>
    simplifySum("flang-simplify-hlfir-sum",
                llvm::cl::desc("Expand hlfir.sum into an inline sequence"),
                llvm::cl::init(false));
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
    mlir::Type elementType = hlfir::getFortranElementType(sum.getType());
    hlfir::Entity array = hlfir::Entity{sum.getArray()};
    mlir::Value mask = sum.getMask();
    mlir::Value dim = sum.getDim();
    bool isTotalReduction = hlfir::Entity{sum}.getRank() == 0;
    int64_t dimVal =
        isTotalReduction ? 0 : fir::getIntIfConstant(dim).value_or(0);
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

      // Create temporary scalar for keeping the running reduction value.
      mlir::Value reductionTemp =
          builder.createTemporaryAlloc(loc, elementType, ".sum.reduction");
      // Initial value for the reduction.
      mlir::Value initValue = genInitValue(loc, builder, elementType);
      builder.create<fir::StoreOp>(loc, initValue, reductionTemp);

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

      // NOTE: the outer elemental operation may be lowered into
      // omp.workshare.loop_wrapper/omp.loop_nest later, so the reduction
      // loop may appear disjoint from the workshare loop nest.
      bool emitWorkshareLoop =
          isTotalReduction ? flangomp::shouldUseWorkshareLowering(sum) : false;

      hlfir::LoopNest loopNest = hlfir::genLoopNest(
          loc, builder, extents, isUnordered, emitWorkshareLoop);

      llvm::SmallVector<mlir::Value> indices;
      if (isTotalReduction) {
        indices = loopNest.oneBasedIndices;
      } else {
        indices = inputIndices;
        indices.insert(indices.begin() + dimVal - 1,
                       loopNest.oneBasedIndices[0]);
      }

      builder.setInsertionPointToStart(loopNest.body);
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
        ifOp = builder.create<fir::IfOp>(loc, isUnmasked,
                                         /*withElseRegion=*/false);

        // In the 'then' block do the actual addition.
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      }

      mlir::Value reductionValue =
          builder.create<fir::LoadOp>(loc, reductionTemp);
      hlfir::Entity element = hlfir::getElementAt(loc, builder, array, indices);
      hlfir::Entity elementValue =
          hlfir::loadTrivialScalar(loc, builder, element);
      // NOTE: we can use "Kahan summation" same way as the runtime
      // (e.g. when fast-math is not allowed), but let's start with
      // the simple version.
      reductionValue = genScalarAdd(loc, builder, reductionValue, elementValue);
      builder.create<fir::StoreOp>(loc, reductionValue, reductionTemp);

      builder.setInsertionPointAfter(loopNest.outerOp);
      return hlfir::Entity{builder.create<fir::LoadOp>(loc, reductionTemp)};
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
      if (!simplifySum)
        return true;

      // Always inline total reductions.
      if (hlfir::Entity{sum}.getRank() == 0)
        return false;
      mlir::Value dim = sum.getDim();
      if (!dim)
        return false;

      if (auto dimVal = fir::getIntIfConstant(dim)) {
        fir::SequenceType arrayTy = mlir::cast<fir::SequenceType>(
            hlfir::getFortranElementOrSequenceType(sum.getArray().getType()));
        if (*dimVal > 0 && *dimVal <= arrayTy.getDimension()) {
          // Ignore SUMs with illegal DIM values.
          // They may appear in dead code,
          // and they do not have to be converted.
          return false;
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
