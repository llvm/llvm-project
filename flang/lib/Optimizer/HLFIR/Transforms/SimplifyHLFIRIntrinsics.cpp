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

// Helper class to generate operations related to computing
// product of values.
class ProductFactory {
public:
  ProductFactory(mlir::Location loc, fir::FirOpBuilder &builder)
      : loc(loc), builder(builder) {}

  // Generate an update of the inner product value:
  //   acc += v1 * v2, OR
  //   acc += CONJ(v1) * v2, OR
  //   acc ||= v1 && v2
  //
  // CONJ parameter specifies whether the first complex product argument
  // needs to be conjugated.
  template <bool CONJ = false>
  mlir::Value genAccumulateProduct(mlir::Value acc, mlir::Value v1,
                                   mlir::Value v2) {
    mlir::Type resultType = acc.getType();
    acc = castToProductType(acc, resultType);
    v1 = castToProductType(v1, resultType);
    v2 = castToProductType(v2, resultType);
    mlir::Value result;
    if (mlir::isa<mlir::FloatType>(resultType)) {
      result = builder.create<mlir::arith::AddFOp>(
          loc, acc, builder.create<mlir::arith::MulFOp>(loc, v1, v2));
    } else if (mlir::isa<mlir::ComplexType>(resultType)) {
      if constexpr (CONJ)
        result = fir::IntrinsicLibrary{builder, loc}.genConjg(resultType, v1);
      else
        result = v1;

      result = builder.create<fir::AddcOp>(
          loc, acc, builder.create<fir::MulcOp>(loc, result, v2));
    } else if (mlir::isa<mlir::IntegerType>(resultType)) {
      result = builder.create<mlir::arith::AddIOp>(
          loc, acc, builder.create<mlir::arith::MulIOp>(loc, v1, v2));
    } else if (mlir::isa<fir::LogicalType>(resultType)) {
      result = builder.create<mlir::arith::OrIOp>(
          loc, acc, builder.create<mlir::arith::AndIOp>(loc, v1, v2));
    } else {
      llvm_unreachable("unsupported type");
    }

    return builder.createConvert(loc, resultType, result);
  }

private:
  mlir::Location loc;
  fir::FirOpBuilder &builder;

  mlir::Value castToProductType(mlir::Value value, mlir::Type type) {
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
      return helper.insertComplexPart(zeroCmplx,
                                      castToProductType(value, partType),
                                      /*isImagPart=*/false);
    }
    return builder.createConvert(loc, type, value);
  }
};

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
    llvm::SmallVector<mlir::Value, 2> inExtents =
        hlfir::genExtentsVector(loc, builder, array);

    // transpose indices
    assert(inExtents.size() == 2 && "checked in TransposeOp::validate");
    return builder.create<fir::ShapeOp>(
        loc, mlir::ValueRange{inExtents[1], inExtents[0]});
  }
};

/// Base class for converting reduction-like operations into
/// a reduction loop[-nest] optionally wrapped into hlfir.elemental.
/// It is used to handle operations produced for ALL, ANY, COUNT,
/// MAXLOC, MAXVAL, MINLOC, MINVAL, SUM intrinsics.
///
/// All of these operations take an input array, and optional
/// dim, mask arguments. ALL, ANY, COUNT do not have mask argument.
class ReductionAsElementalConverter {
public:
  ReductionAsElementalConverter(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter)
      : op{op}, rewriter{rewriter}, loc{op->getLoc()}, builder{rewriter, op} {
    assert(op->getNumResults() == 1);
  }
  virtual ~ReductionAsElementalConverter() {}

  /// Do the actual conversion or return mlir::failure(),
  /// if conversion is not possible.
  mlir::LogicalResult convert();

private:
  // Return fir.shape specifying the shape of the result
  // of a reduction with DIM=dimVal. The second return value
  // is the extent of the DIM dimension.
  std::tuple<mlir::Value, mlir::Value>
  genResultShapeForPartialReduction(hlfir::Entity array, int64_t dimVal);

  /// \p mask is a scalar or array logical mask.
  /// If \p isPresentPred is not nullptr, it is a dynamic predicate value
  /// identifying whether the mask's variable is present.
  /// \p indices is a range of one-based indices to access \p mask
  /// when it is an array.
  ///
  /// The method returns the scalar mask value to guard the access
  /// to a single element of the input array.
  mlir::Value genMaskValue(mlir::Value mask, mlir::Value isPresentPred,
                           mlir::ValueRange indices);

protected:
  /// Return the input array.
  virtual mlir::Value getSource() const = 0;

  /// Return DIM or nullptr, if it is not present.
  virtual mlir::Value getDim() const = 0;

  /// Return MASK or nullptr, if it is not present.
  virtual mlir::Value getMask() const { return nullptr; }

  /// Return FastMathFlags attached to the operation
  /// or arith::FastMathFlags::none, if the operation
  /// does not support FastMathFlags (e.g. ALL, ANY, COUNT).
  virtual mlir::arith::FastMathFlags getFastMath() const {
    return mlir::arith::FastMathFlags::none;
  }

  /// Generates initial values for the reduction values used
  /// by the reduction loop. In general, there is a single
  /// loop-carried reduction value (e.g. for SUM), but, for example,
  /// MAXLOC/MINLOC implementation uses multiple reductions.
  virtual llvm::SmallVector<mlir::Value> genReductionInitValues() = 0;

  /// Perform reduction(s) update given a single input array's element
  /// identified by \p array and \p oneBasedIndices coordinates.
  /// \p currentValue specifies the current value(s) of the reduction(s)
  /// inside the reduction loop body.
  virtual llvm::SmallVector<mlir::Value>
  reduceOneElement(const llvm::SmallVectorImpl<mlir::Value> &currentValue,
                   hlfir::Entity array, mlir::ValueRange oneBasedIndices) = 0;

  /// Given reduction value(s) in \p reductionResults produced
  /// by the reduction loop, apply any required updates and return
  /// new reduction value(s) to be used after the reduction loop
  /// (e.g. as the result yield of the wrapping hlfir.elemental).
  /// NOTE: if the reduction loop is wrapped in hlfir.elemental,
  /// the insertion point of any generated code is inside hlfir.elemental.
  virtual hlfir::Entity
  genFinalResult(const llvm::SmallVectorImpl<mlir::Value> &reductionResults) {
    assert(reductionResults.size() == 1 &&
           "default implementation of genFinalResult expect a single reduction "
           "value");
    return hlfir::Entity{reductionResults[0]};
  }

  /// Return mlir::success(), if the operation can be converted.
  /// The default implementation always returns mlir::success().
  /// The derived type may override the default implementation
  /// with its own definition.
  virtual mlir::LogicalResult isConvertible() const { return mlir::success(); }

  // Default implementation of isTotalReduction() just checks
  // if the result of the operation is a scalar.
  // True result indicates that the reduction has to be done
  // across all elements, false result indicates that
  // the result is an array expression produced by an hlfir.elemental
  // operation with a single reduction loop across the DIM dimension.
  //
  // MAXLOC/MINLOC must override this.
  virtual bool isTotalReduction() const { return getResultRank() == 0; }

  // Return true, if the reduction loop[-nest] may be unordered.
  // In general, FP reductions may only be unordered when
  // FastMathFlags::reassoc transformations are allowed.
  //
  // Some dervied types may need to override this.
  virtual bool isUnordered() const {
    mlir::Type elemType = getSourceElementType();
    if (mlir::isa<mlir::IntegerType, fir::LogicalType, fir::CharacterType>(
            elemType))
      return true;
    return static_cast<bool>(getFastMath() &
                             mlir::arith::FastMathFlags::reassoc);
  }

  /// Return 0, if DIM is not present or its values does not matter
  /// (for example, a reduction of 1D array does not care about
  /// the DIM value, assuming that it is a valid program).
  /// Return mlir::failure(), if DIM is a constant known
  /// to be invalid for the given array.
  /// Otherwise, return DIM constant value.
  mlir::FailureOr<int64_t> getConstDim() const {
    int64_t dimVal = 0;
    if (!isTotalReduction()) {
      // In case of partial reduction we should ignore the operations
      // with invalid DIM values. They may appear in dead code
      // after constant propagation.
      auto constDim = fir::getIntIfConstant(getDim());
      if (!constDim)
        return rewriter.notifyMatchFailure(op, "Nonconstant DIM");
      dimVal = *constDim;

      if ((dimVal <= 0 || dimVal > getSourceRank()))
        return rewriter.notifyMatchFailure(op,
                                           "Invalid DIM for partial reduction");
    }
    return dimVal;
  }

  /// Return hlfir::Entity of the result.
  hlfir::Entity getResultEntity() const {
    return hlfir::Entity{op->getResult(0)};
  }

  /// Return type of the result (e.g. !hlfir.expr<?xi32>).
  mlir::Type getResultType() const { return getResultEntity().getType(); }

  /// Return the element type of the result (e.g. i32).
  mlir::Type getResultElementType() const {
    return hlfir::getFortranElementType(getResultType());
  }

  /// Return rank of the result.
  unsigned getResultRank() const { return getResultEntity().getRank(); }

  /// Return the element type of the source.
  mlir::Type getSourceElementType() const {
    return hlfir::getFortranElementType(getSource().getType());
  }

  /// Return rank of the input array.
  unsigned getSourceRank() const {
    return hlfir::Entity{getSource()}.getRank();
  }

  /// The reduction operation.
  mlir::Operation *op;

  mlir::PatternRewriter &rewriter;
  mlir::Location loc;
  fir::FirOpBuilder builder;
};

/// Generate initialization value for MIN or MAX reduction
/// of the given \p type.
template <bool IS_MAX>
static mlir::Value genMinMaxInitValue(mlir::Location loc,
                                      fir::FirOpBuilder &builder,
                                      mlir::Type type) {
  if (auto ty = mlir::dyn_cast<mlir::FloatType>(type)) {
    const llvm::fltSemantics &sem = ty.getFloatSemantics();
    // We must not use +/-INF here. If the reduction input is empty,
    // the result of reduction must be +/-LARGEST.
    llvm::APFloat limit = llvm::APFloat::getLargest(sem, /*Negative=*/IS_MAX);
    return builder.createRealConstant(loc, type, limit);
  }
  unsigned bits = type.getIntOrFloatBitWidth();
  int64_t limitInt = IS_MAX
                         ? llvm::APInt::getSignedMinValue(bits).getSExtValue()
                         : llvm::APInt::getSignedMaxValue(bits).getSExtValue();
  return builder.createIntegerConstant(loc, type, limitInt);
}

/// Generate a comparison of an array element value \p elem
/// and the current reduction value \p reduction for MIN/MAX reduction.
template <bool IS_MAX>
static mlir::Value
genMinMaxComparison(mlir::Location loc, fir::FirOpBuilder &builder,
                    mlir::Value elem, mlir::Value reduction) {
  if (mlir::isa<mlir::FloatType>(reduction.getType())) {
    // For FP reductions we want the first smallest value to be used, that
    // is not NaN. A OGL/OLT condition will usually work for this unless all
    // the values are Nan or Inf. This follows the same logic as
    // NumericCompare for Minloc/Maxloc in extrema.cpp.
    mlir::Value cmp = builder.create<mlir::arith::CmpFOp>(
        loc,
        IS_MAX ? mlir::arith::CmpFPredicate::OGT
               : mlir::arith::CmpFPredicate::OLT,
        elem, reduction);
    mlir::Value cmpNan = builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::UNE, reduction, reduction);
    mlir::Value cmpNan2 = builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OEQ, elem, elem);
    cmpNan = builder.create<mlir::arith::AndIOp>(loc, cmpNan, cmpNan2);
    return builder.create<mlir::arith::OrIOp>(loc, cmp, cmpNan);
  } else if (mlir::isa<mlir::IntegerType>(reduction.getType())) {
    return builder.create<mlir::arith::CmpIOp>(
        loc,
        IS_MAX ? mlir::arith::CmpIPredicate::sgt
               : mlir::arith::CmpIPredicate::slt,
        elem, reduction);
  }
  llvm_unreachable("unsupported type");
}

/// Implementation of ReductionAsElementalConverter interface
/// for MAXLOC/MINLOC.
template <typename T>
class MinMaxlocAsElementalConverter : public ReductionAsElementalConverter {
  static_assert(std::is_same_v<T, hlfir::MaxlocOp> ||
                std::is_same_v<T, hlfir::MinlocOp>);
  static constexpr unsigned maxRank = Fortran::common::maxRank;
  // We have the following reduction values in the reduction loop:
  //   * N integer coordinates, where N is:
  //     - RANK(ARRAY) for total reductions.
  //     - 1 for partial reductions.
  //   * 1 reduction value holding the current MIN/MAX.
  //   * 1 boolean indicating whether it is the first time
  //     the mask is true.
  static constexpr unsigned maxNumReductions = Fortran::common::maxRank + 2;
  static constexpr bool isMax = std::is_same_v<T, hlfir::MaxlocOp>;
  using Base = ReductionAsElementalConverter;

public:
  MinMaxlocAsElementalConverter(T op, mlir::PatternRewriter &rewriter)
      : Base{op.getOperation(), rewriter} {}

private:
  virtual mlir::Value getSource() const final { return getOp().getArray(); }
  virtual mlir::Value getDim() const final { return getOp().getDim(); }
  virtual mlir::Value getMask() const final { return getOp().getMask(); }
  virtual mlir::arith::FastMathFlags getFastMath() const final {
    return getOp().getFastmath();
  }

  virtual mlir::LogicalResult isConvertible() const final {
    if (getOp().getBack())
      return rewriter.notifyMatchFailure(
          getOp(), "BACK is not supported for MINLOC/MAXLOC inlining");
    if (mlir::isa<fir::CharacterType>(getSourceElementType()))
      return rewriter.notifyMatchFailure(
          getOp(),
          "CHARACTER type is not supported for MINLOC/MAXLOC inlining");
    return mlir::success();
  }

  // If the result is scalar, then DIM does not matter,
  // and this is a total reduction.
  // If DIM is not present, this is a total reduction.
  virtual bool isTotalReduction() const final {
    return getResultRank() == 0 || !getDim();
  }

  virtual llvm::SmallVector<mlir::Value> genReductionInitValues() final;
  virtual llvm::SmallVector<mlir::Value>
  reduceOneElement(const llvm::SmallVectorImpl<mlir::Value> &currentValue,
                   hlfir::Entity array, mlir::ValueRange oneBasedIndices) final;
  virtual hlfir::Entity genFinalResult(
      const llvm::SmallVectorImpl<mlir::Value> &reductionResults) final;

private:
  T getOp() const { return mlir::cast<T>(op); }

  unsigned getNumCoors() const {
    return isTotalReduction() ? getSourceRank() : 1;
  }

  void
  checkReductions(const llvm::SmallVectorImpl<mlir::Value> &reductions) const {
    assert(reductions.size() == getNumCoors() + 2 &&
           "invalid number of reductions for MINLOC/MAXLOC");
  }

  mlir::Value
  getCurrentMinMax(const llvm::SmallVectorImpl<mlir::Value> &reductions) const {
    checkReductions(reductions);
    return reductions[getNumCoors()];
  }

  mlir::Value
  getIsFirst(const llvm::SmallVectorImpl<mlir::Value> &reductions) const {
    checkReductions(reductions);
    return reductions[getNumCoors() + 1];
  }
};

template <typename T>
llvm::SmallVector<mlir::Value>
MinMaxlocAsElementalConverter<T>::genReductionInitValues() {
  // Initial value for the coordinate(s) is zero.
  mlir::Value zeroCoor =
      fir::factory::createZeroValue(builder, loc, getResultElementType());
  llvm::SmallVector<mlir::Value> result(getNumCoors(), zeroCoor);

  // Initial value for the MIN/MAX value.
  mlir::Value minMaxInit =
      genMinMaxInitValue<isMax>(loc, builder, getSourceElementType());
  result.push_back(minMaxInit);

  // Initial value for isFirst predicate. It is switched to false,
  // when the reduction update dynamically happens inside the reduction
  // loop.
  mlir::Value trueVal = builder.createBool(loc, true);
  result.push_back(trueVal);

  return result;
}

template <typename T>
llvm::SmallVector<mlir::Value>
MinMaxlocAsElementalConverter<T>::reduceOneElement(
    const llvm::SmallVectorImpl<mlir::Value> &currentValue, hlfir::Entity array,
    mlir::ValueRange oneBasedIndices) {
  checkReductions(currentValue);
  hlfir::Entity elementValue =
      hlfir::loadElementAt(loc, builder, array, oneBasedIndices);
  mlir::Value cmp = genMinMaxComparison<isMax>(loc, builder, elementValue,
                                               getCurrentMinMax(currentValue));
  // If isFirst is true, then do the reduction update regardless
  // of the FP comparison.
  cmp = builder.create<mlir::arith::OrIOp>(loc, cmp, getIsFirst(currentValue));

  llvm::SmallVector<mlir::Value> newIndices;
  int64_t dim = 1;
  if (!isTotalReduction()) {
    auto dimVal = getConstDim();
    assert(mlir::succeeded(dimVal) &&
           "partial MINLOC/MAXLOC reduction with invalid DIM");
    dim = *dimVal;
    assert(getNumCoors() == 1 &&
           "partial MAXLOC/MINLOC reduction must compute one coordinate");
  }

  for (unsigned coorIdx = 0; coorIdx < getNumCoors(); ++coorIdx) {
    mlir::Value currentCoor = currentValue[coorIdx];
    mlir::Value newCoor = builder.createConvert(
        loc, currentCoor.getType(), oneBasedIndices[coorIdx + dim - 1]);
    mlir::Value update =
        builder.create<mlir::arith::SelectOp>(loc, cmp, newCoor, currentCoor);
    newIndices.push_back(update);
  }

  mlir::Value newMinMax = builder.create<mlir::arith::SelectOp>(
      loc, cmp, elementValue, getCurrentMinMax(currentValue));
  newIndices.push_back(newMinMax);

  mlir::Value newIsFirst = builder.createBool(loc, false);
  newIndices.push_back(newIsFirst);

  assert(currentValue.size() == newIndices.size() &&
         "invalid number of updated reductions");

  return newIndices;
}

template <typename T>
hlfir::Entity MinMaxlocAsElementalConverter<T>::genFinalResult(
    const llvm::SmallVectorImpl<mlir::Value> &reductionResults) {
  // Identification of the final result of MINLOC/MAXLOC:
  //   * If DIM is absent, the result is rank-one array.
  //   * If DIM is present:
  //     - The result is scalar for rank-one input.
  //     - The result is an array of rank RANK(ARRAY)-1.
  checkReductions(reductionResults);

  // We need to adjust the one-based indices to real array indices.
  // The adjustment must only be done, if there was an actual update
  // of the coordinates in the reduction loop. For this check we only
  // need to compare if any of the reduction results is not zero.
  mlir::Value zero = fir::factory::createZeroValue(
      builder, loc, reductionResults[0].getType());
  mlir::Value doAdjust = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, reductionResults[0], zero);
  mlir::Type indexType = builder.getIndexType();
  mlir::Value one = builder.createIntegerConstant(loc, indexType, 1);

  auto adjustCoor = [&](mlir::Value coor, mlir::Value lbound) {
    assert(mlir::isa<mlir::IndexType>(lbound.getType()));
    mlir::Value coorAsIndex = builder.createConvert(loc, indexType, coor);
    mlir::Value tmp =
        builder.create<mlir::arith::AddIOp>(loc, coorAsIndex, lbound);
    tmp = builder.create<mlir::arith::SubIOp>(loc, tmp, one);
    tmp =
        builder.create<mlir::arith::SelectOp>(loc, doAdjust, tmp, coorAsIndex);
    return builder.createConvert(loc, coor.getType(), tmp);
  };

  // For partial reductions, the final result of the reduction
  // loop is just a scalar - the coordinate within DIM dimension.
  if (getResultRank() == 0 || !isTotalReduction()) {
    // The result is a scalar, so just return the scalar.
    assert(getNumCoors() == 1 &&
           "unpexpected number of coordinates for scalar result");

    int64_t dim = 1;
    if (!isTotalReduction()) {
      auto dimVal = getConstDim();
      assert(mlir::succeeded(dimVal) &&
             "partial MINLOC/MAXLOC reduction with invalid DIM");
      dim = *dimVal;
    }
    mlir::Value dimLbound =
        hlfir::genLBound(loc, builder, hlfir::Entity{getSource()}, dim - 1);
    return hlfir::Entity{adjustCoor(reductionResults[0], dimLbound)};
  }
  // This is a total reduction, and there is no wrapping hlfir.elemental.
  // We have to pack the reduced coordinates into a rank-one array.
  unsigned rank = getSourceRank();
  // TODO: in order to avoid introducing new memory effects
  // we should not use a temporary in memory.
  // We can use hlfir.elemental with a switch to pack all the coordinates
  // into an array expression, or we can have a dedicated HLFIR operation
  // for this.
  mlir::Value tempArray = builder.createTemporary(
      loc, fir::SequenceType::get(rank, getResultElementType()));
  llvm::SmallVector<mlir::Value, maxRank> arrayLbounds =
      hlfir::genLBounds(loc, builder, hlfir::Entity(getSource()));
  for (unsigned i = 0; i < rank; ++i) {
    mlir::Value coor = adjustCoor(reductionResults[i], arrayLbounds[i]);
    mlir::Value idx = builder.createIntegerConstant(loc, indexType, i + 1);
    mlir::Value resultElement =
        hlfir::getElementAt(loc, builder, hlfir::Entity{tempArray}, {idx});
    builder.create<hlfir::AssignOp>(loc, coor, resultElement);
  }
  mlir::Value tempExpr = builder.create<hlfir::AsExprOp>(
      loc, tempArray, builder.createBool(loc, false));
  return hlfir::Entity{tempExpr};
}

/// Base class for numeric reductions like MAXVAl, MINVAL, SUM.
template <typename OpT>
class NumericReductionAsElementalConverterBase
    : public ReductionAsElementalConverter {
  using Base = ReductionAsElementalConverter;

protected:
  NumericReductionAsElementalConverterBase(OpT op,
                                           mlir::PatternRewriter &rewriter)
      : Base{op.getOperation(), rewriter} {}

  virtual mlir::Value getSource() const final { return getOp().getArray(); }
  virtual mlir::Value getDim() const final { return getOp().getDim(); }
  virtual mlir::Value getMask() const final { return getOp().getMask(); }
  virtual mlir::arith::FastMathFlags getFastMath() const final {
    return getOp().getFastmath();
  }

  OpT getOp() const { return mlir::cast<OpT>(op); }

  void checkReductions(const llvm::SmallVectorImpl<mlir::Value> &reductions) {
    assert(reductions.size() == 1 && "reduction must produce single value");
  }
};

/// Reduction converter for MAXMAL/MINVAL.
template <typename T>
class MinMaxvalAsElementalConverter
    : public NumericReductionAsElementalConverterBase<T> {
  static_assert(std::is_same_v<T, hlfir::MaxvalOp> ||
                std::is_same_v<T, hlfir::MinvalOp>);
  static constexpr bool isMax = std::is_same_v<T, hlfir::MaxvalOp>;
  using Base = NumericReductionAsElementalConverterBase<T>;

public:
  MinMaxvalAsElementalConverter(T op, mlir::PatternRewriter &rewriter)
      : Base{op, rewriter} {}

private:
  virtual mlir::LogicalResult isConvertible() const final {
    if (mlir::isa<fir::CharacterType>(this->getSourceElementType()))
      return this->rewriter.notifyMatchFailure(
          this->getOp(),
          "CHARACTER type is not supported for MINVAL/MAXVAL inlining");
    return mlir::success();
  }

  virtual llvm::SmallVector<mlir::Value> genReductionInitValues() final {
    return {genMinMaxInitValue<isMax>(this->loc, this->builder,
                                      this->getResultElementType())};
  }
  virtual llvm::SmallVector<mlir::Value>
  reduceOneElement(const llvm::SmallVectorImpl<mlir::Value> &currentValue,
                   hlfir::Entity array,
                   mlir::ValueRange oneBasedIndices) final {
    this->checkReductions(currentValue);
    fir::FirOpBuilder &builder = this->builder;
    mlir::Location loc = this->loc;
    hlfir::Entity elementValue =
        hlfir::loadElementAt(loc, builder, array, oneBasedIndices);
    mlir::Value cmp =
        genMinMaxComparison<isMax>(loc, builder, elementValue, currentValue[0]);
    return {builder.create<mlir::arith::SelectOp>(loc, cmp, elementValue,
                                                  currentValue[0])};
  }
};

/// Reduction converter for SUM.
class SumAsElementalConverter
    : public NumericReductionAsElementalConverterBase<hlfir::SumOp> {
  using Base = NumericReductionAsElementalConverterBase;

public:
  SumAsElementalConverter(hlfir::SumOp op, mlir::PatternRewriter &rewriter)
      : Base{op, rewriter} {}

private:
  virtual llvm::SmallVector<mlir::Value> genReductionInitValues() final {
    return {
        fir::factory::createZeroValue(builder, loc, getResultElementType())};
  }
  virtual llvm::SmallVector<mlir::Value>
  reduceOneElement(const llvm::SmallVectorImpl<mlir::Value> &currentValue,
                   hlfir::Entity array,
                   mlir::ValueRange oneBasedIndices) final {
    checkReductions(currentValue);
    hlfir::Entity elementValue =
        hlfir::loadElementAt(loc, builder, array, oneBasedIndices);
    // NOTE: we can use "Kahan summation" same way as the runtime
    // (e.g. when fast-math is not allowed), but let's start with
    // the simple version.
    return {genScalarAdd(currentValue[0], elementValue)};
  }

  // Generate scalar addition of the two values (of the same data type).
  mlir::Value genScalarAdd(mlir::Value value1, mlir::Value value2);
};

/// Base class for logical reductions like ALL, ANY, COUNT.
/// They do not have MASK and FastMathFlags.
template <typename OpT>
class LogicalReductionAsElementalConverterBase
    : public ReductionAsElementalConverter {
  using Base = ReductionAsElementalConverter;

public:
  LogicalReductionAsElementalConverterBase(OpT op,
                                           mlir::PatternRewriter &rewriter)
      : Base{op.getOperation(), rewriter} {}

protected:
  OpT getOp() const { return mlir::cast<OpT>(op); }

  void checkReductions(const llvm::SmallVectorImpl<mlir::Value> &reductions) {
    assert(reductions.size() == 1 && "reduction must produce single value");
  }

  virtual mlir::Value getSource() const final { return getOp().getMask(); }
  virtual mlir::Value getDim() const final { return getOp().getDim(); }

  virtual hlfir::Entity genFinalResult(
      const llvm::SmallVectorImpl<mlir::Value> &reductionResults) override {
    checkReductions(reductionResults);
    return hlfir::Entity{reductionResults[0]};
  }
};

/// Reduction converter for ALL/ANY.
template <typename T>
class AllAnyAsElementalConverter
    : public LogicalReductionAsElementalConverterBase<T> {
  static_assert(std::is_same_v<T, hlfir::AllOp> ||
                std::is_same_v<T, hlfir::AnyOp>);
  static constexpr bool isAll = std::is_same_v<T, hlfir::AllOp>;
  using Base = LogicalReductionAsElementalConverterBase<T>;

public:
  AllAnyAsElementalConverter(T op, mlir::PatternRewriter &rewriter)
      : Base{op, rewriter} {}

private:
  virtual llvm::SmallVector<mlir::Value> genReductionInitValues() final {
    return {this->builder.createBool(this->loc, isAll ? true : false)};
  }
  virtual llvm::SmallVector<mlir::Value>
  reduceOneElement(const llvm::SmallVectorImpl<mlir::Value> &currentValue,
                   hlfir::Entity array,
                   mlir::ValueRange oneBasedIndices) final {
    this->checkReductions(currentValue);
    fir::FirOpBuilder &builder = this->builder;
    mlir::Location loc = this->loc;
    hlfir::Entity elementValue =
        hlfir::loadElementAt(loc, builder, array, oneBasedIndices);
    mlir::Value mask =
        builder.createConvert(loc, builder.getI1Type(), elementValue);
    if constexpr (isAll)
      return {builder.create<mlir::arith::AndIOp>(loc, mask, currentValue[0])};
    else
      return {builder.create<mlir::arith::OrIOp>(loc, mask, currentValue[0])};
  }

  virtual hlfir::Entity genFinalResult(
      const llvm::SmallVectorImpl<mlir::Value> &reductionValues) final {
    this->checkReductions(reductionValues);
    return hlfir::Entity{this->builder.createConvert(
        this->loc, this->getResultElementType(), reductionValues[0])};
  }
};

/// Reduction converter for COUNT.
class CountAsElementalConverter
    : public LogicalReductionAsElementalConverterBase<hlfir::CountOp> {
  using Base = LogicalReductionAsElementalConverterBase<hlfir::CountOp>;

public:
  CountAsElementalConverter(hlfir::CountOp op, mlir::PatternRewriter &rewriter)
      : Base{op, rewriter} {}

private:
  virtual llvm::SmallVector<mlir::Value> genReductionInitValues() final {
    return {
        fir::factory::createZeroValue(builder, loc, getResultElementType())};
  }
  virtual llvm::SmallVector<mlir::Value>
  reduceOneElement(const llvm::SmallVectorImpl<mlir::Value> &currentValue,
                   hlfir::Entity array,
                   mlir::ValueRange oneBasedIndices) final {
    checkReductions(currentValue);
    hlfir::Entity elementValue =
        hlfir::loadElementAt(loc, builder, array, oneBasedIndices);
    mlir::Value cond =
        builder.createConvert(loc, builder.getI1Type(), elementValue);
    mlir::Value one =
        builder.createIntegerConstant(loc, getResultElementType(), 1);
    mlir::Value add1 =
        builder.create<mlir::arith::AddIOp>(loc, currentValue[0], one);
    return {builder.create<mlir::arith::SelectOp>(loc, cond, add1,
                                                  currentValue[0])};
  }
};

mlir::LogicalResult ReductionAsElementalConverter::convert() {
  mlir::LogicalResult canConvert(isConvertible());

  if (mlir::failed(canConvert))
    return canConvert;

  hlfir::Entity array = hlfir::Entity{getSource()};
  bool isTotalReduce = isTotalReduction();
  auto dimVal = getConstDim();
  if (mlir::failed(dimVal))
    return dimVal;
  mlir::Value mask = getMask();
  mlir::Value resultShape, dimExtent;
  llvm::SmallVector<mlir::Value> arrayExtents;
  if (isTotalReduce)
    arrayExtents = hlfir::genExtentsVector(loc, builder, array);
  else
    std::tie(resultShape, dimExtent) =
        genResultShapeForPartialReduction(array, *dimVal);

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
      maskValue = genMaskValue(mask, isPresentPred, {});
  }

  auto genKernel = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                       mlir::ValueRange inputIndices) -> hlfir::Entity {
    // Loop over all indices in the DIM dimension, and reduce all values.
    // If DIM is not present, do total reduction.

    // Initial value for the reduction.
    llvm::SmallVector<mlir::Value, 1> reductionInitValues =
        genReductionInitValues();

    llvm::SmallVector<mlir::Value> extents;
    if (isTotalReduce)
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
      if (isTotalReduce) {
        indices = oneBasedIndices;
      } else {
        indices = inputIndices;
        indices.insert(indices.begin() + *dimVal - 1, oneBasedIndices[0]);
      }

      llvm::SmallVector<mlir::Value, 1> reductionValues = reductionArgs;
      llvm::SmallVector<mlir::Type, 1> reductionTypes;
      llvm::transform(reductionValues, std::back_inserter(reductionTypes),
                      [](mlir::Value v) { return v.getType(); });
      fir::IfOp ifOp;
      if (mask) {
        // Make the reduction value update conditional on the value
        // of the mask.
        if (!maskValue) {
          // If the mask is an array, use the elemental and the loop indices
          // to address the proper mask element.
          maskValue = genMaskValue(mask, isPresentPred, indices);
        }
        mlir::Value isUnmasked =
            builder.create<fir::ConvertOp>(loc, builder.getI1Type(), maskValue);
        ifOp = builder.create<fir::IfOp>(loc, reductionTypes, isUnmasked,
                                         /*withElseRegion=*/true);
        // In the 'else' block return the current reduction value.
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        builder.create<fir::ResultOp>(loc, reductionValues);

        // In the 'then' block do the actual addition.
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      }
      reductionValues = reduceOneElement(reductionValues, array, indices);
      if (ifOp) {
        builder.create<fir::ResultOp>(loc, reductionValues);
        builder.setInsertionPointAfter(ifOp);
        reductionValues = ifOp.getResults();
      }

      return reductionValues;
    };

    llvm::SmallVector<mlir::Value, 1> reductionFinalValues =
        hlfir::genLoopNestWithReductions(
            loc, builder, extents, reductionInitValues, genBody, isUnordered());
    return genFinalResult(reductionFinalValues);
  };

  if (isTotalReduce) {
    hlfir::Entity result = genKernel(loc, builder, mlir::ValueRange{});
    rewriter.replaceOp(op, result);
    return mlir::success();
  }

  hlfir::ElementalOp elementalOp = hlfir::genElementalOp(
      loc, builder, getResultElementType(), resultShape, /*typeParams=*/{},
      genKernel,
      /*isUnordered=*/true, /*polymorphicMold=*/nullptr, getResultType());

  // it wouldn't be safe to replace block arguments with a different
  // hlfir.expr type. Types can differ due to differing amounts of shape
  // information
  assert(elementalOp.getResult().getType() == op->getResult(0).getType());

  rewriter.replaceOp(op, elementalOp);
  return mlir::success();
}

std::tuple<mlir::Value, mlir::Value>
ReductionAsElementalConverter::genResultShapeForPartialReduction(
    hlfir::Entity array, int64_t dimVal) {
  llvm::SmallVector<mlir::Value> inExtents =
      hlfir::genExtentsVector(loc, builder, array);
  assert(dimVal > 0 && dimVal <= static_cast<int64_t>(inExtents.size()) &&
         "DIM must be present and a positive constant not exceeding "
         "the array's rank");

  mlir::Value dimExtent = inExtents[dimVal - 1];
  inExtents.erase(inExtents.begin() + dimVal - 1);
  return {builder.create<fir::ShapeOp>(loc, inExtents), dimExtent};
}

mlir::Value SumAsElementalConverter::genScalarAdd(mlir::Value value1,
                                                  mlir::Value value2) {
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

mlir::Value ReductionAsElementalConverter::genMaskValue(
    mlir::Value mask, mlir::Value isPresentPred, mlir::ValueRange indices) {
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

/// Convert an operation that is a partial or total reduction
/// over an array of values into a reduction loop[-nest]
/// optionally wrapped into hlfir.elemental.
template <typename Op>
class ReductionConversion : public mlir::OpRewritePattern<Op> {
public:
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    if constexpr (std::is_same_v<Op, hlfir::MaxlocOp> ||
                  std::is_same_v<Op, hlfir::MinlocOp>) {
      MinMaxlocAsElementalConverter<Op> converter(op, rewriter);
      return converter.convert();
    } else if constexpr (std::is_same_v<Op, hlfir::MaxvalOp> ||
                         std::is_same_v<Op, hlfir::MinvalOp>) {
      MinMaxvalAsElementalConverter<Op> converter(op, rewriter);
      return converter.convert();
    } else if constexpr (std::is_same_v<Op, hlfir::CountOp>) {
      CountAsElementalConverter converter(op, rewriter);
      return converter.convert();
    } else if constexpr (std::is_same_v<Op, hlfir::AllOp> ||
                         std::is_same_v<Op, hlfir::AnyOp>) {
      AllAnyAsElementalConverter<Op> converter(op, rewriter);
      return converter.convert();
    } else if constexpr (std::is_same_v<Op, hlfir::SumOp>) {
      SumAsElementalConverter converter{op, rewriter};
      return converter.convert();
    }
    return rewriter.notifyMatchFailure(op, "unexpected reduction operation");
  }
};

class CShiftConversion : public mlir::OpRewritePattern<hlfir::CShiftOp> {
public:
  using mlir::OpRewritePattern<hlfir::CShiftOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::CShiftOp cshift,
                  mlir::PatternRewriter &rewriter) const override {

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

    // When DIM==1 and the contiguity of the input array is not statically
    // known, try to exploit the fact that the leading dimension might be
    // contiguous. We can do this now using hlfir.eval_in_mem with
    // a dynamic check for the leading dimension contiguity.
    // Otherwise, convert hlfir.cshift to hlfir.elemental.
    //
    // Note that the hlfir.elemental can be inlined into other hlfir.elemental,
    // while hlfir.eval_in_mem prevents this, and we will end up creating
    // a temporary array for the result. We may need to come up with
    // a more sophisticated logic for picking the most efficient
    // representation.
    hlfir::Entity array = hlfir::Entity{cshift.getArray()};
    mlir::Type elementType = array.getFortranElementType();
    if (dimVal == 1 && fir::isa_trivial(elementType) &&
        // genInMemCShift() only works for variables currently.
        array.isVariable())
      rewriter.replaceOp(cshift, genInMemCShift(rewriter, cshift, dimVal));
    else
      rewriter.replaceOp(cshift, genElementalCShift(rewriter, cshift, dimVal));
    return mlir::success();
  }

private:
  /// Generate MODULO(\p shiftVal, \p extent).
  static mlir::Value normalizeShiftValue(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         mlir::Value shiftVal,
                                         mlir::Value extent,
                                         mlir::Type calcType) {
    shiftVal = builder.createConvert(loc, calcType, shiftVal);
    extent = builder.createConvert(loc, calcType, extent);
    // Make sure that we do not divide by zero. When the dimension
    // has zero size, turn the extent into 1. Note that the computed
    // MODULO value won't be used in this case, so it does not matter
    // which extent value we use.
    mlir::Value zero = builder.createIntegerConstant(loc, calcType, 0);
    mlir::Value one = builder.createIntegerConstant(loc, calcType, 1);
    mlir::Value isZero = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, extent, zero);
    extent = builder.create<mlir::arith::SelectOp>(loc, isZero, one, extent);
    shiftVal = fir::IntrinsicLibrary{builder, loc}.genModulo(
        calcType, {shiftVal, extent});
    return builder.createConvert(loc, calcType, shiftVal);
  }

  /// Convert \p cshift into an hlfir.elemental using
  /// the pre-computed constant \p dimVal.
  static mlir::Operation *genElementalCShift(mlir::PatternRewriter &rewriter,
                                             hlfir::CShiftOp cshift,
                                             int64_t dimVal) {
    using Fortran::common::maxRank;
    hlfir::Entity shift = hlfir::Entity{cshift.getShift()};
    hlfir::Entity array = hlfir::Entity{cshift.getArray()};

    mlir::Location loc = cshift.getLoc();
    fir::FirOpBuilder builder{rewriter, cshift.getOperation()};
    // The new index computation involves MODULO, which is not implemented
    // for IndexType, so use I64 instead.
    mlir::Type calcType = builder.getI64Type();
    // All the indices arithmetic used below does not overflow
    // signed and unsigned I64.
    builder.setIntegerOverflowFlags(mlir::arith::IntegerOverflowFlags::nsw |
                                    mlir::arith::IntegerOverflowFlags::nuw);

    mlir::Value arrayShape = hlfir::genShape(loc, builder, array);
    llvm::SmallVector<mlir::Value, maxRank> arrayExtents =
        hlfir::getExplicitExtentsFromShape(arrayShape, builder);
    llvm::SmallVector<mlir::Value, 1> typeParams;
    hlfir::genLengthParameters(loc, builder, array, typeParams);
    mlir::Value shiftDimExtent =
        builder.createConvert(loc, calcType, arrayExtents[dimVal - 1]);
    mlir::Value shiftVal;
    if (shift.isScalar()) {
      shiftVal = hlfir::loadTrivialScalar(loc, builder, shift);
      shiftVal =
          normalizeShiftValue(loc, builder, shiftVal, shiftDimExtent, calcType);
    }

    auto genKernel = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                         mlir::ValueRange inputIndices) -> hlfir::Entity {
      llvm::SmallVector<mlir::Value, maxRank> indices{inputIndices};
      if (!shiftVal) {
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
        shiftVal = normalizeShiftValue(loc, builder, shiftVal, shiftDimExtent,
                                       calcType);
      }

      // Element i of the result (1-based) is element
      // 'MODULO(i + SH - 1, SIZE(ARRAY,DIM)) + 1' (1-based) of the original
      // ARRAY (or its section, when ARRAY is not a vector).

      // Compute the index into the original array using the normalized
      // shift value, which satisfies (SH >= 0 && SH < SIZE(ARRAY,DIM)):
      //   newIndex =
      //     i + ((i <= SIZE(ARRAY,DIM) - SH) ? SH : SH - SIZE(ARRAY,DIM))
      //
      // Such index computation allows for further loop vectorization
      // in LLVM.
      mlir::Value wrapBound =
          builder.create<mlir::arith::SubIOp>(loc, shiftDimExtent, shiftVal);
      mlir::Value adjustedShiftVal =
          builder.create<mlir::arith::SubIOp>(loc, shiftVal, shiftDimExtent);
      mlir::Value index =
          builder.createConvert(loc, calcType, inputIndices[dimVal - 1]);
      mlir::Value wrapCheck = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sle, index, wrapBound);
      mlir::Value actualShift = builder.create<mlir::arith::SelectOp>(
          loc, wrapCheck, shiftVal, adjustedShiftVal);
      mlir::Value newIndex =
          builder.create<mlir::arith::AddIOp>(loc, index, actualShift);
      newIndex = builder.createConvert(loc, builder.getIndexType(), newIndex);
      indices[dimVal - 1] = newIndex;
      hlfir::Entity element = hlfir::getElementAt(loc, builder, array, indices);
      return hlfir::loadTrivialScalar(loc, builder, element);
    };

    mlir::Type elementType = array.getFortranElementType();
    hlfir::ElementalOp elementalOp = hlfir::genElementalOp(
        loc, builder, elementType, arrayShape, typeParams, genKernel,
        /*isUnordered=*/true,
        array.isPolymorphic() ? static_cast<mlir::Value>(array) : nullptr,
        cshift.getResult().getType());
    return elementalOp.getOperation();
  }

  /// Convert \p cshift into an hlfir.eval_in_mem using the pre-computed
  /// constant \p dimVal.
  /// The converted code looks like this:
  ///   do i=1,SH
  ///     result(i + (SIZE(ARRAY,DIM) - SH)) = array(i)
  ///   end
  ///   do i=1,SIZE(ARRAY,DIM) - SH
  ///     result(i) = array(i + SH)
  ///   end
  ///
  /// When \p dimVal is 1, we generate the same code twice
  /// under a dynamic check for the contiguity of the leading
  /// dimension. In the code corresponding to the contiguous
  /// leading dimension, the shift dimension is represented
  /// as a contiguous slice of the original array.
  /// This allows recognizing the above two loops as memcpy
  /// loop idioms in LLVM.
  static mlir::Operation *genInMemCShift(mlir::PatternRewriter &rewriter,
                                         hlfir::CShiftOp cshift,
                                         int64_t dimVal) {
    using Fortran::common::maxRank;
    hlfir::Entity shift = hlfir::Entity{cshift.getShift()};
    hlfir::Entity array = hlfir::Entity{cshift.getArray()};
    assert(array.isVariable() && "array must be a variable");
    assert(!array.isPolymorphic() &&
           "genInMemCShift does not support polymorphic types");
    mlir::Location loc = cshift.getLoc();
    fir::FirOpBuilder builder{rewriter, cshift.getOperation()};
    // The new index computation involves MODULO, which is not implemented
    // for IndexType, so use I64 instead.
    mlir::Type calcType = builder.getI64Type();
    // All the indices arithmetic used below does not overflow
    // signed and unsigned I64.
    builder.setIntegerOverflowFlags(mlir::arith::IntegerOverflowFlags::nsw |
                                    mlir::arith::IntegerOverflowFlags::nuw);

    mlir::Value arrayShape = hlfir::genShape(loc, builder, array);
    llvm::SmallVector<mlir::Value, maxRank> arrayExtents =
        hlfir::getExplicitExtentsFromShape(arrayShape, builder);
    llvm::SmallVector<mlir::Value, 1> typeParams;
    hlfir::genLengthParameters(loc, builder, array, typeParams);
    mlir::Value shiftDimExtent =
        builder.createConvert(loc, calcType, arrayExtents[dimVal - 1]);
    mlir::Value shiftVal;
    if (shift.isScalar()) {
      shiftVal = hlfir::loadTrivialScalar(loc, builder, shift);
      shiftVal =
          normalizeShiftValue(loc, builder, shiftVal, shiftDimExtent, calcType);
    }

    hlfir::EvaluateInMemoryOp evalOp =
        builder.create<hlfir::EvaluateInMemoryOp>(
            loc, mlir::cast<hlfir::ExprType>(cshift.getType()), arrayShape);
    builder.setInsertionPointToStart(&evalOp.getBody().front());

    mlir::Value resultArray = evalOp.getMemory();
    mlir::Type arrayType = fir::dyn_cast_ptrEleTy(resultArray.getType());
    resultArray = builder.createBox(loc, fir::BoxType::get(arrayType),
                                    resultArray, arrayShape, /*slice=*/nullptr,
                                    typeParams, /*tdesc=*/nullptr);

    // This is a generator of the dimension shift code.
    // The code is inserted inside a loop nest over the other dimensions
    // (if any). If exposeContiguity is true, the array's section
    // array(s(1), ..., s(dim-1), :, s(dim+1), ..., s(n)) is represented
    // as a contiguous 1D array.
    // shiftVal is the normalized shift value that satisfies (SH >= 0 && SH <
    // SIZE(ARRAY,DIM)).
    //
    auto genDimensionShift = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                                 mlir::Value shiftVal, bool exposeContiguity,
                                 mlir::ValueRange oneBasedIndices)
        -> llvm::SmallVector<mlir::Value, 0> {
      // Create a vector of indices (s(1), ..., s(dim-1), nullptr, s(dim+1),
      // ..., s(n)) so that we can update the dimVal index as needed.
      llvm::SmallVector<mlir::Value, maxRank> srcIndices(
          oneBasedIndices.begin(), oneBasedIndices.begin() + (dimVal - 1));
      srcIndices.push_back(nullptr);
      srcIndices.append(oneBasedIndices.begin() + (dimVal - 1),
                        oneBasedIndices.end());
      llvm::SmallVector<mlir::Value, maxRank> dstIndices(srcIndices);

      hlfir::Entity srcArray = array;
      if (exposeContiguity && mlir::isa<fir::BaseBoxType>(srcArray.getType())) {
        assert(dimVal == 1 && "can expose contiguity only for dim 1");
        llvm::SmallVector<mlir::Value, maxRank> arrayLbounds =
            hlfir::genLowerbounds(loc, builder, arrayShape, array.getRank());
        hlfir::Entity section =
            hlfir::gen1DSection(loc, builder, srcArray, dimVal, arrayLbounds,
                                arrayExtents, oneBasedIndices, typeParams);
        mlir::Value addr = hlfir::genVariableRawAddress(loc, builder, section);
        mlir::Value shape = hlfir::genShape(loc, builder, section);
        mlir::Type boxType = fir::wrapInClassOrBoxType(
            hlfir::getFortranElementOrSequenceType(section.getType()),
            section.isPolymorphic());
        srcArray = hlfir::Entity{
            builder.createBox(loc, boxType, addr, shape, /*slice=*/nullptr,
                              /*lengths=*/{}, /*tdesc=*/nullptr)};
        // When shifting the dimension as a 1D section of the original
        // array, we only need one index for addressing.
        srcIndices.resize(1);
      }

      // Copy first portion of the array:
      // do i=1,SH
      //   result(i + (SIZE(ARRAY,DIM) - SH)) = array(i)
      // end
      auto genAssign1 = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                            mlir::ValueRange index,
                            mlir::ValueRange reductionArgs)
          -> llvm::SmallVector<mlir::Value, 0> {
        assert(index.size() == 1 && "expected single loop");
        mlir::Value srcIndex = builder.createConvert(loc, calcType, index[0]);
        srcIndices[dimVal - 1] = srcIndex;
        hlfir::Entity srcElementValue =
            hlfir::loadElementAt(loc, builder, srcArray, srcIndices);
        mlir::Value dstIndex = builder.create<mlir::arith::AddIOp>(
            loc, srcIndex,
            builder.create<mlir::arith::SubIOp>(loc, shiftDimExtent, shiftVal));
        dstIndices[dimVal - 1] = dstIndex;
        hlfir::Entity dstElement = hlfir::getElementAt(
            loc, builder, hlfir::Entity{resultArray}, dstIndices);
        builder.create<hlfir::AssignOp>(loc, srcElementValue, dstElement);
        return {};
      };

      // Generate the first loop.
      hlfir::genLoopNestWithReductions(loc, builder, {shiftVal},
                                       /*reductionInits=*/{}, genAssign1,
                                       /*isUnordered=*/true);

      // Copy second portion of the array:
      // do i=1,SIZE(ARRAY,DIM)-SH
      //   result(i) = array(i + SH)
      // end
      auto genAssign2 = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                            mlir::ValueRange index,
                            mlir::ValueRange reductionArgs)
          -> llvm::SmallVector<mlir::Value, 0> {
        assert(index.size() == 1 && "expected single loop");
        mlir::Value dstIndex = builder.createConvert(loc, calcType, index[0]);
        mlir::Value srcIndex =
            builder.create<mlir::arith::AddIOp>(loc, dstIndex, shiftVal);
        srcIndices[dimVal - 1] = srcIndex;
        hlfir::Entity srcElementValue =
            hlfir::loadElementAt(loc, builder, srcArray, srcIndices);
        dstIndices[dimVal - 1] = dstIndex;
        hlfir::Entity dstElement = hlfir::getElementAt(
            loc, builder, hlfir::Entity{resultArray}, dstIndices);
        builder.create<hlfir::AssignOp>(loc, srcElementValue, dstElement);
        return {};
      };

      // Generate the second loop.
      mlir::Value bound =
          builder.create<mlir::arith::SubIOp>(loc, shiftDimExtent, shiftVal);
      hlfir::genLoopNestWithReductions(loc, builder, {bound},
                                       /*reductionInits=*/{}, genAssign2,
                                       /*isUnordered=*/true);
      return {};
    };

    // A wrapper around genDimensionShift that computes the normalized
    // shift value and manages the insertion of the multiple versions
    // of the shift based on the dynamic check of the leading dimension's
    // contiguity (when dimVal == 1).
    auto genShiftBody = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                            mlir::ValueRange oneBasedIndices,
                            mlir::ValueRange reductionArgs)
        -> llvm::SmallVector<mlir::Value, 0> {
      // Copy the dimension with a shift:
      // SH is either SHIFT (if scalar) or SHIFT(oneBasedIndices).
      if (!shiftVal) {
        assert(!oneBasedIndices.empty() && "scalar shift must be precomputed");
        hlfir::Entity shiftElement =
            hlfir::getElementAt(loc, builder, shift, oneBasedIndices);
        shiftVal = hlfir::loadTrivialScalar(loc, builder, shiftElement);
        shiftVal = normalizeShiftValue(loc, builder, shiftVal, shiftDimExtent,
                                       calcType);
      }

      // If we can fetch the byte stride of the leading dimension,
      // and the byte size of the element, then we can generate
      // a dynamic contiguity check and expose the leading dimension's
      // contiguity in FIR, making memcpy loop idiom recognition
      // possible.
      mlir::Value elemSize;
      mlir::Value stride;
      if (dimVal == 1 && mlir::isa<fir::BaseBoxType>(array.getType())) {
        mlir::Type indexType = builder.getIndexType();
        elemSize =
            builder.create<fir::BoxEleSizeOp>(loc, indexType, array.getBase());
        mlir::Value dimIdx =
            builder.createIntegerConstant(loc, indexType, dimVal - 1);
        auto boxDim = builder.create<fir::BoxDimsOp>(
            loc, indexType, indexType, indexType, array.getBase(), dimIdx);
        stride = boxDim.getByteStride();
      }

      if (array.isSimplyContiguous() || !elemSize || !stride) {
        genDimensionShift(loc, builder, shiftVal, /*exposeContiguity=*/false,
                          oneBasedIndices);
        return {};
      }

      mlir::Value isContiguous = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, elemSize, stride);
      builder.genIfOp(loc, {}, isContiguous, /*withElseRegion=*/true)
          .genThen([&]() {
            genDimensionShift(loc, builder, shiftVal, /*exposeContiguity=*/true,
                              oneBasedIndices);
          })
          .genElse([&]() {
            genDimensionShift(loc, builder, shiftVal,
                              /*exposeContiguity=*/false, oneBasedIndices);
          });

      return {};
    };

    // For 1D case, generate a single loop.
    // For ND case, generate a loop nest over the other dimensions
    // with a single loop inside (generated separately).
    llvm::SmallVector<mlir::Value, maxRank> newExtents(arrayExtents);
    newExtents.erase(newExtents.begin() + (dimVal - 1));
    if (!newExtents.empty())
      hlfir::genLoopNestWithReductions(loc, builder, newExtents,
                                       /*reductionInits=*/{}, genShiftBody,
                                       /*isUnordered=*/true);
    else
      genShiftBody(loc, builder, {}, {});

    return evalOp.getOperation();
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
    llvm::SmallVector<mlir::Value, 2> input1Extents =
        hlfir::genExtentsVector(loc, builder, input1);
    llvm::SmallVector<mlir::Value, 2> input2Extents =
        hlfir::genExtentsVector(loc, builder, input2);

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
        mlir::Value productValue =
            ProductFactory{loc, builder}.genAccumulateProduct(
                resultElementValue, lhsElementValue, rhsElementValue);
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
        mlir::Value productValue =
            ProductFactory{loc, builder}.genAccumulateProduct(
                resultElementValue, lhsElementValue, rhsElementValue);
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
        mlir::Value productValue =
            ProductFactory{loc, builder}.genAccumulateProduct(
                resultElementValue, lhsElementValue, rhsElementValue);
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
        mlir::Value productValue =
            ProductFactory{loc, builder}.genAccumulateProduct(
                reductionArgs[0], lhsElementValue, rhsElementValue);
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

class DotProductConversion
    : public mlir::OpRewritePattern<hlfir::DotProductOp> {
public:
  using mlir::OpRewritePattern<hlfir::DotProductOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::DotProductOp product,
                  mlir::PatternRewriter &rewriter) const override {
    hlfir::Entity op = hlfir::Entity{product};
    if (!op.isScalar())
      return rewriter.notifyMatchFailure(product, "produces non-scalar result");

    mlir::Location loc = product.getLoc();
    fir::FirOpBuilder builder{rewriter, product.getOperation()};
    hlfir::Entity lhs = hlfir::Entity{product.getLhs()};
    hlfir::Entity rhs = hlfir::Entity{product.getRhs()};
    mlir::Type resultElementType = product.getType();
    bool isUnordered = mlir::isa<mlir::IntegerType>(resultElementType) ||
                       mlir::isa<fir::LogicalType>(resultElementType) ||
                       static_cast<bool>(builder.getFastMathFlags() &
                                         mlir::arith::FastMathFlags::reassoc);

    mlir::Value extent = genProductExtent(loc, builder, lhs, rhs);

    auto genBody = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                       mlir::ValueRange oneBasedIndices,
                       mlir::ValueRange reductionArgs)
        -> llvm::SmallVector<mlir::Value, 1> {
      hlfir::Entity lhsElementValue =
          hlfir::loadElementAt(loc, builder, lhs, oneBasedIndices);
      hlfir::Entity rhsElementValue =
          hlfir::loadElementAt(loc, builder, rhs, oneBasedIndices);
      mlir::Value productValue =
          ProductFactory{loc, builder}.genAccumulateProduct</*CONJ=*/true>(
              reductionArgs[0], lhsElementValue, rhsElementValue);
      return {productValue};
    };

    mlir::Value initValue =
        fir::factory::createZeroValue(builder, loc, resultElementType);

    llvm::SmallVector<mlir::Value, 1> result = hlfir::genLoopNestWithReductions(
        loc, builder, {extent},
        /*reductionInits=*/{initValue}, genBody, isUnordered);

    rewriter.replaceOp(product, result[0]);
    return mlir::success();
  }

private:
  static mlir::Value genProductExtent(mlir::Location loc,
                                      fir::FirOpBuilder &builder,
                                      hlfir::Entity input1,
                                      hlfir::Entity input2) {
    llvm::SmallVector<mlir::Value, 1> input1Extents =
        hlfir::genExtentsVector(loc, builder, input1);
    llvm::SmallVector<mlir::Value, 1> input2Extents =
        hlfir::genExtentsVector(loc, builder, input2);

    assert(input1Extents.size() == 1 && input2Extents.size() == 1 &&
           "hlfir.dot_product arguments must be vectors");
    llvm::SmallVector<mlir::Value, 1> extent =
        fir::factory::deduceOptimalExtents(input1Extents, input2Extents);
    return extent[0];
  }
};

class ReshapeAsElementalConversion
    : public mlir::OpRewritePattern<hlfir::ReshapeOp> {
public:
  using mlir::OpRewritePattern<hlfir::ReshapeOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(hlfir::ReshapeOp reshape,
                  mlir::PatternRewriter &rewriter) const override {
    // Do not inline RESHAPE with ORDER yet. The runtime implementation
    // may be good enough, unless the temporary creation overhead
    // is high.
    // TODO: If ORDER is constant, then we can still easily inline.
    // TODO: If the result's rank is 1, then we can assume ORDER == (/1/).
    if (reshape.getOrder())
      return rewriter.notifyMatchFailure(reshape,
                                         "RESHAPE with ORDER argument");

    // Verify that the element types of ARRAY, PAD and the result
    // match before doing any transformations. For example,
    // the character types of different lengths may appear in the dead
    // code, and it just does not make sense to inline hlfir.reshape
    // in this case (a runtime call might have less code size footprint).
    hlfir::Entity result = hlfir::Entity{reshape};
    hlfir::Entity array = hlfir::Entity{reshape.getArray()};
    mlir::Type elementType = array.getFortranElementType();
    if (result.getFortranElementType() != elementType)
      return rewriter.notifyMatchFailure(
          reshape, "ARRAY and result have different types");
    mlir::Value pad = reshape.getPad();
    if (pad && hlfir::getFortranElementType(pad.getType()) != elementType)
      return rewriter.notifyMatchFailure(reshape,
                                         "ARRAY and PAD have different types");

    // TODO: selecting between ARRAY and PAD of non-trivial element types
    // requires more work. We have to select between two references
    // to elements in ARRAY and PAD. This requires conditional
    // bufferization of the element, if ARRAY/PAD is an expression.
    if (pad && !fir::isa_trivial(elementType))
      return rewriter.notifyMatchFailure(reshape,
                                         "PAD present with non-trivial type");

    mlir::Location loc = reshape.getLoc();
    fir::FirOpBuilder builder{rewriter, reshape.getOperation()};
    // Assume that all the indices arithmetic does not overflow
    // the IndexType.
    builder.setIntegerOverflowFlags(mlir::arith::IntegerOverflowFlags::nuw);

    llvm::SmallVector<mlir::Value, 1> typeParams;
    hlfir::genLengthParameters(loc, builder, array, typeParams);

    // Fetch the extents of ARRAY, PAD and result beforehand.
    llvm::SmallVector<mlir::Value, Fortran::common::maxRank> arrayExtents =
        hlfir::genExtentsVector(loc, builder, array);

    // If PAD is present, we have to use array size to start taking
    // elements from the PAD array.
    mlir::Value arraySize =
        pad ? computeArraySize(loc, builder, arrayExtents) : nullptr;
    hlfir::Entity shape = hlfir::Entity{reshape.getShape()};
    llvm::SmallVector<mlir::Value, Fortran::common::maxRank> resultExtents;
    mlir::Type indexType = builder.getIndexType();
    for (int idx = 0; idx < result.getRank(); ++idx)
      resultExtents.push_back(hlfir::loadElementAt(
          loc, builder, shape,
          builder.createIntegerConstant(loc, indexType, idx + 1)));
    auto resultShape = builder.create<fir::ShapeOp>(loc, resultExtents);

    auto genKernel = [&](mlir::Location loc, fir::FirOpBuilder &builder,
                         mlir::ValueRange inputIndices) -> hlfir::Entity {
      mlir::Value linearIndex =
          computeLinearIndex(loc, builder, resultExtents, inputIndices);
      fir::IfOp ifOp;
      if (pad) {
        // PAD is present. Check if this element comes from the PAD array.
        mlir::Value isInsideArray = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::ult, linearIndex, arraySize);
        ifOp = builder.create<fir::IfOp>(loc, elementType, isInsideArray,
                                         /*withElseRegion=*/true);

        // In the 'else' block, return an element from the PAD.
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        // PAD is dynamically optional, but we can unconditionally access it
        // in the 'else' block. If we have to start taking elements from it,
        // then it must be present in a valid program.
        llvm::SmallVector<mlir::Value, Fortran::common::maxRank> padExtents =
            hlfir::genExtentsVector(loc, builder, hlfir::Entity{pad});
        // Subtract the ARRAY size from the zero-based linear index
        // to get the zero-based linear index into PAD.
        mlir::Value padLinearIndex =
            builder.create<mlir::arith::SubIOp>(loc, linearIndex, arraySize);
        llvm::SmallVector<mlir::Value, Fortran::common::maxRank> padIndices =
            delinearizeIndex(loc, builder, padExtents, padLinearIndex,
                             /*wrapAround=*/true);
        mlir::Value padElement =
            hlfir::loadElementAt(loc, builder, hlfir::Entity{pad}, padIndices);
        builder.create<fir::ResultOp>(loc, padElement);

        // In the 'then' block, return an element from the ARRAY.
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      }

      llvm::SmallVector<mlir::Value, Fortran::common::maxRank> arrayIndices =
          delinearizeIndex(loc, builder, arrayExtents, linearIndex,
                           /*wrapAround=*/false);
      mlir::Value arrayElement =
          hlfir::loadElementAt(loc, builder, array, arrayIndices);

      if (ifOp) {
        builder.create<fir::ResultOp>(loc, arrayElement);
        builder.setInsertionPointAfter(ifOp);
        arrayElement = ifOp.getResult(0);
      }

      return hlfir::Entity{arrayElement};
    };
    hlfir::ElementalOp elementalOp = hlfir::genElementalOp(
        loc, builder, elementType, resultShape, typeParams, genKernel,
        /*isUnordered=*/true,
        /*polymorphicMold=*/result.isPolymorphic() ? array : mlir::Value{},
        reshape.getResult().getType());
    assert(elementalOp.getResult().getType() == reshape.getResult().getType());
    rewriter.replaceOp(reshape, elementalOp);
    return mlir::success();
  }

private:
  /// Compute zero-based linear index given an array extents
  /// and one-based indices:
  ///   \p extents: [e0, e1, ..., en]
  ///   \p indices: [i0, i1, ..., in]
  ///
  /// linear-index :=
  ///   (...((in-1)*e(n-1)+(i(n-1)-1))*e(n-2)+...)*e0+(i0-1)
  static mlir::Value computeLinearIndex(mlir::Location loc,
                                        fir::FirOpBuilder &builder,
                                        mlir::ValueRange extents,
                                        mlir::ValueRange indices) {
    std::size_t rank = extents.size();
    assert(rank == indices.size());
    mlir::Type indexType = builder.getIndexType();
    mlir::Value zero = builder.createIntegerConstant(loc, indexType, 0);
    mlir::Value one = builder.createIntegerConstant(loc, indexType, 1);
    mlir::Value linearIndex = zero;
    std::size_t idx = 0;
    for (auto index : llvm::reverse(indices)) {
      mlir::Value tmp = builder.create<mlir::arith::SubIOp>(
          loc, builder.createConvert(loc, indexType, index), one);
      tmp = builder.create<mlir::arith::AddIOp>(loc, linearIndex, tmp);
      if (idx + 1 < rank)
        tmp = builder.create<mlir::arith::MulIOp>(
            loc, tmp,
            builder.createConvert(loc, indexType, extents[rank - idx - 2]));

      linearIndex = tmp;
      ++idx;
    }
    return linearIndex;
  }

  /// Compute one-based array indices from the given zero-based \p linearIndex
  /// and the array \p extents [e0, e1, ..., en].
  ///   i0 := linearIndex % e0 + 1
  ///   linearIndex := linearIndex / e0
  ///   i1 := linearIndex % e1 + 1
  ///   linearIndex := linearIndex / e1
  ///   ...
  ///   i(n-1) := linearIndex % e(n-1) + 1
  ///   linearIndex := linearIndex / e(n-1)
  ///   if (wrapAround) {
  ///     // If the index is allowed to wrap around, then
  ///     // we need to modulo it by the last dimension's extent.
  ///     in := linearIndex % en + 1
  ///   } else {
  ///     in := linearIndex + 1
  ///   }
  static llvm::SmallVector<mlir::Value, Fortran::common::maxRank>
  delinearizeIndex(mlir::Location loc, fir::FirOpBuilder &builder,
                   mlir::ValueRange extents, mlir::Value linearIndex,
                   bool wrapAround) {
    llvm::SmallVector<mlir::Value, Fortran::common::maxRank> indices;
    mlir::Type indexType = builder.getIndexType();
    mlir::Value one = builder.createIntegerConstant(loc, indexType, 1);
    linearIndex = builder.createConvert(loc, indexType, linearIndex);

    for (std::size_t dim = 0; dim < extents.size(); ++dim) {
      mlir::Value extent = builder.createConvert(loc, indexType, extents[dim]);
      // Avoid the modulo for the last index, unless wrap around is allowed.
      mlir::Value currentIndex = linearIndex;
      if (dim != extents.size() - 1 || wrapAround)
        currentIndex =
            builder.create<mlir::arith::RemUIOp>(loc, linearIndex, extent);
      // The result of the last division is unused, so it will be DCEd.
      linearIndex =
          builder.create<mlir::arith::DivUIOp>(loc, linearIndex, extent);
      indices.push_back(
          builder.create<mlir::arith::AddIOp>(loc, currentIndex, one));
    }
    return indices;
  }

  /// Return size of an array given its extents.
  static mlir::Value computeArraySize(mlir::Location loc,
                                      fir::FirOpBuilder &builder,
                                      mlir::ValueRange extents) {
    mlir::Type indexType = builder.getIndexType();
    mlir::Value size = builder.createIntegerConstant(loc, indexType, 1);
    for (auto extent : extents)
      size = builder.create<mlir::arith::MulIOp>(
          loc, size, builder.createConvert(loc, indexType, extent));
    return size;
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
    patterns.insert<ReductionConversion<hlfir::SumOp>>(context);
    patterns.insert<CShiftConversion>(context);
    patterns.insert<MatmulConversion<hlfir::MatmulTransposeOp>>(context);

    patterns.insert<ReductionConversion<hlfir::CountOp>>(context);
    patterns.insert<ReductionConversion<hlfir::AnyOp>>(context);
    patterns.insert<ReductionConversion<hlfir::AllOp>>(context);
    patterns.insert<ReductionConversion<hlfir::MaxlocOp>>(context);
    patterns.insert<ReductionConversion<hlfir::MinlocOp>>(context);
    patterns.insert<ReductionConversion<hlfir::MaxvalOp>>(context);
    patterns.insert<ReductionConversion<hlfir::MinvalOp>>(context);

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

    patterns.insert<DotProductConversion>(context);
    patterns.insert<ReshapeAsElementalConversion>(context);

    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      mlir::emitError(getOperation()->getLoc(),
                      "failure in HLFIR intrinsic simplification");
      signalPassFailure();
    }
  }
};
} // namespace
