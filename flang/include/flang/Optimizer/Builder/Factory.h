//===-- Optimizer/Builder/Factory.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Templates to generate more complex code patterns in transformation passes.
// In transformation passes, front-end information such as is available in
// lowering is not available.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_FACTORY_H
#define FORTRAN_OPTIMIZER_BUILDER_FACTORY_H

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/iterator_range.h"

namespace mlir {
class Location;
class Value;
} // namespace mlir

namespace fir::factory {

constexpr llvm::StringRef attrFortranArrayOffsets() {
  return "Fortran.offsets";
}

/// Generate a character copy with optimized forms.
///
/// If the lengths are constant and equal, use load/store rather than a loop.
/// Otherwise, if the lengths are constant and the input is longer than the
/// output, generate a loop to move a truncated portion of the source to the
/// destination. Finally, if the lengths are runtime values or the destination
/// is longer than the source, move the entire source character and pad the
/// destination with spaces as needed.
template <typename B>
void genCharacterCopy(mlir::Value src, mlir::Value srcLen, mlir::Value dst,
                      mlir::Value dstLen, B &builder, mlir::Location loc) {
  auto srcTy =
      fir::dyn_cast_ptrEleTy(src.getType()).template cast<fir::CharacterType>();
  auto dstTy =
      fir::dyn_cast_ptrEleTy(dst.getType()).template cast<fir::CharacterType>();
  if (!srcLen && !dstLen && srcTy.getFKind() == dstTy.getFKind() &&
      srcTy.getLen() == dstTy.getLen()) {
    // same size, so just use load and store
    auto load = builder.template create<fir::LoadOp>(loc, src);
    builder.template create<fir::StoreOp>(loc, load, dst);
    return;
  }
  auto zero = builder.template create<mlir::arith::ConstantIndexOp>(loc, 0);
  auto one = builder.template create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto toArrayTy = [&](fir::CharacterType ty) {
    return fir::ReferenceType::get(fir::SequenceType::get(
        fir::SequenceType::ShapeRef{fir::SequenceType::getUnknownExtent()},
        fir::CharacterType::getSingleton(ty.getContext(), ty.getFKind())));
  };
  auto toEleTy = [&](fir::ReferenceType ty) {
    auto seqTy = ty.getEleTy().cast<fir::SequenceType>();
    return seqTy.getEleTy().cast<fir::CharacterType>();
  };
  auto toCoorTy = [&](fir::ReferenceType ty) {
    return fir::ReferenceType::get(toEleTy(ty));
  };
  if (!srcLen && !dstLen && srcTy.getLen() >= dstTy.getLen()) {
    auto upper = builder.template create<mlir::arith::ConstantIndexOp>(
        loc, dstTy.getLen() - 1);
    auto loop = builder.template create<fir::DoLoopOp>(loc, zero, upper, one);
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(loop.getBody());
    auto csrcTy = toArrayTy(srcTy);
    auto csrc = builder.template create<fir::ConvertOp>(loc, csrcTy, src);
    auto in = builder.template create<fir::CoordinateOp>(
        loc, toCoorTy(csrcTy), csrc, loop.getInductionVar());
    auto load = builder.template create<fir::LoadOp>(loc, in);
    auto cdstTy = toArrayTy(dstTy);
    auto cdst = builder.template create<fir::ConvertOp>(loc, cdstTy, dst);
    auto out = builder.template create<fir::CoordinateOp>(
        loc, toCoorTy(cdstTy), cdst, loop.getInductionVar());
    mlir::Value cast =
        srcTy.getFKind() == dstTy.getFKind()
            ? load.getResult()
            : builder
                  .template create<fir::ConvertOp>(loc, toEleTy(cdstTy), load)
                  .getResult();
    builder.template create<fir::StoreOp>(loc, cast, out);
    builder.restoreInsertionPoint(insPt);
    return;
  }
  auto minusOne = [&](mlir::Value v) -> mlir::Value {
    return builder.template create<mlir::arith::SubIOp>(
        loc, builder.template create<fir::ConvertOp>(loc, one.getType(), v),
        one);
  };
  mlir::Value len = dstLen ? minusOne(dstLen)
                           : builder
                                 .template create<mlir::arith::ConstantIndexOp>(
                                     loc, dstTy.getLen() - 1)
                                 .getResult();
  auto loop = builder.template create<fir::DoLoopOp>(loc, zero, len, one);
  auto insPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(loop.getBody());
  mlir::Value slen =
      srcLen
          ? builder.template create<fir::ConvertOp>(loc, one.getType(), srcLen)
                .getResult()
          : builder
                .template create<mlir::arith::ConstantIndexOp>(loc,
                                                               srcTy.getLen())
                .getResult();
  auto cond = builder.template create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, loop.getInductionVar(), slen);
  auto ifOp = builder.template create<fir::IfOp>(loc, cond, /*withElse=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto csrcTy = toArrayTy(srcTy);
  auto csrc = builder.template create<fir::ConvertOp>(loc, csrcTy, src);
  auto in = builder.template create<fir::CoordinateOp>(
      loc, toCoorTy(csrcTy), csrc, loop.getInductionVar());
  auto load = builder.template create<fir::LoadOp>(loc, in);
  auto cdstTy = toArrayTy(dstTy);
  auto cdst = builder.template create<fir::ConvertOp>(loc, cdstTy, dst);
  auto out = builder.template create<fir::CoordinateOp>(
      loc, toCoorTy(cdstTy), cdst, loop.getInductionVar());
  mlir::Value cast =
      srcTy.getFKind() == dstTy.getFKind()
          ? load.getResult()
          : builder.template create<fir::ConvertOp>(loc, toEleTy(cdstTy), load)
                .getResult();
  builder.template create<fir::StoreOp>(loc, cast, out);
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  auto space = builder.template create<fir::StringLitOp>(
      loc, toEleTy(cdstTy), llvm::ArrayRef<char>{' '});
  auto cdst2 = builder.template create<fir::ConvertOp>(loc, cdstTy, dst);
  auto out2 = builder.template create<fir::CoordinateOp>(
      loc, toCoorTy(cdstTy), cdst2, loop.getInductionVar());
  builder.template create<fir::StoreOp>(loc, space, out2);
  builder.restoreInsertionPoint(insPt);
}

/// Get extents from fir.shape/fir.shape_shift op. Empty result if
/// \p shapeVal is empty or is a fir.shift.
inline llvm::SmallVector<mlir::Value> getExtents(mlir::Value shapeVal) {
  if (shapeVal)
    if (auto *shapeOp = shapeVal.getDefiningOp()) {
      if (auto shOp = mlir::dyn_cast<fir::ShapeOp>(shapeOp)) {
        auto operands = shOp.getExtents();
        return {operands.begin(), operands.end()};
      }
      if (auto shOp = mlir::dyn_cast<fir::ShapeShiftOp>(shapeOp)) {
        auto operands = shOp.getExtents();
        return {operands.begin(), operands.end()};
      }
    }
  return {};
}

/// Get origins from fir.shape_shift/fir.shift op. Empty result if
/// \p shapeVal is empty or is a fir.shape.
inline llvm::SmallVector<mlir::Value> getOrigins(mlir::Value shapeVal) {
  if (shapeVal)
    if (auto *shapeOp = shapeVal.getDefiningOp()) {
      if (auto shOp = mlir::dyn_cast<fir::ShapeShiftOp>(shapeOp)) {
        auto operands = shOp.getOrigins();
        return {operands.begin(), operands.end()};
      }
      if (auto shOp = mlir::dyn_cast<fir::ShiftOp>(shapeOp)) {
        auto operands = shOp.getOrigins();
        return {operands.begin(), operands.end()};
      }
    }
  return {};
}

/// Convert the normalized indices on array_fetch and array_update to the
/// dynamic (and non-zero) origin required by array_coor.
/// Do not adjust any trailing components in the path as they specify a
/// particular path into the array value and must already correspond to the
/// structure of an element.
template <typename B>
llvm::SmallVector<mlir::Value>
originateIndices(mlir::Location loc, B &builder, mlir::Type memTy,
                 mlir::Value shapeVal, mlir::ValueRange indices) {
  llvm::SmallVector<mlir::Value> result;
  auto origins = getOrigins(shapeVal);
  if (origins.empty()) {
    assert(!shapeVal || mlir::isa<fir::ShapeOp>(shapeVal.getDefiningOp()));
    auto ty = fir::dyn_cast_ptrOrBoxEleTy(memTy);
    assert(ty && ty.isa<fir::SequenceType>());
    auto seqTy = ty.cast<fir::SequenceType>();
    auto one = builder.template create<mlir::arith::ConstantIndexOp>(loc, 1);
    const auto dimension = seqTy.getDimension();
    if (shapeVal) {
      assert(dimension == mlir::cast<fir::ShapeOp>(shapeVal.getDefiningOp())
                              .getType()
                              .getRank());
    }
    for (auto i : llvm::enumerate(indices)) {
      if (i.index() < dimension) {
        assert(fir::isa_integer(i.value().getType()));
        result.push_back(
            builder.template create<mlir::arith::AddIOp>(loc, i.value(), one));
      } else {
        result.push_back(i.value());
      }
    }
    return result;
  }
  const auto dimension = origins.size();
  unsigned origOff = 0;
  for (auto i : llvm::enumerate(indices)) {
    if (i.index() < dimension)
      result.push_back(builder.template create<mlir::arith::AddIOp>(
          loc, i.value(), origins[origOff++]));
    else
      result.push_back(i.value());
  }
  return result;
}

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_FACTORY_H
