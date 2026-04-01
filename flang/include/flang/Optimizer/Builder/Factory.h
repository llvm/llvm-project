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
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/iterator_range.h"

namespace aiir {
class Location;
class Value;
} // namespace aiir

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
void genCharacterCopy(aiir::Value src, aiir::Value srcLen, aiir::Value dst,
                      aiir::Value dstLen, B &builder, aiir::Location loc) {
  auto srcTy =
      aiir::cast<fir::CharacterType>(fir::dyn_cast_ptrEleTy(src.getType()));
  auto dstTy =
      aiir::cast<fir::CharacterType>(fir::dyn_cast_ptrEleTy(dst.getType()));
  if (!srcLen && !dstLen && srcTy.getFKind() == dstTy.getFKind() &&
      srcTy.getLen() == dstTy.getLen()) {
    // same size, so just use load and store
    auto load = fir::LoadOp::create(builder, loc, src);
    fir::StoreOp::create(builder, loc, load, dst);
    return;
  }
  auto zero = aiir::arith::ConstantIndexOp::create(builder, loc, 0);
  auto one = aiir::arith::ConstantIndexOp::create(builder, loc, 1);
  auto toArrayTy = [&](fir::CharacterType ty) {
    return fir::ReferenceType::get(fir::SequenceType::get(
        fir::SequenceType::ShapeRef{fir::SequenceType::getUnknownExtent()},
        fir::CharacterType::getSingleton(ty.getContext(), ty.getFKind())));
  };
  auto toEleTy = [&](fir::ReferenceType ty) {
    auto seqTy = aiir::cast<fir::SequenceType>(ty.getEleTy());
    return aiir::cast<fir::CharacterType>(seqTy.getEleTy());
  };
  auto toCoorTy = [&](fir::ReferenceType ty) {
    return fir::ReferenceType::get(toEleTy(ty));
  };
  if (!srcLen && !dstLen && srcTy.getLen() >= dstTy.getLen()) {
    auto upper =
        aiir::arith::ConstantIndexOp::create(builder, loc, dstTy.getLen() - 1);
    auto loop = fir::DoLoopOp::create(builder, loc, zero, upper, one);
    auto insPt = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(loop.getBody());
    auto csrcTy = toArrayTy(srcTy);
    auto csrc = fir::ConvertOp::create(builder, loc, csrcTy, src);
    auto in = fir::CoordinateOp::create(builder, loc, toCoorTy(csrcTy), csrc,
                                        loop.getInductionVar());
    auto load = fir::LoadOp::create(builder, loc, in);
    auto cdstTy = toArrayTy(dstTy);
    auto cdst = fir::ConvertOp::create(builder, loc, cdstTy, dst);
    auto out = fir::CoordinateOp::create(builder, loc, toCoorTy(cdstTy), cdst,
                                         loop.getInductionVar());
    aiir::Value cast =
        srcTy.getFKind() == dstTy.getFKind()
            ? load.getResult()
            : fir::ConvertOp::create(builder, loc, toEleTy(cdstTy), load)
                  .getResult();
    fir::StoreOp::create(builder, loc, cast, out);
    builder.restoreInsertionPoint(insPt);
    return;
  }
  auto minusOne = [&](aiir::Value v) -> aiir::Value {
    return aiir::arith::SubIOp::create(
        builder, loc, fir::ConvertOp::create(builder, loc, one.getType(), v),
        one);
  };
  aiir::Value len = dstLen ? minusOne(dstLen)
                           : aiir::arith::ConstantIndexOp::create(
                                 builder, loc, dstTy.getLen() - 1)
                                 .getResult();
  auto loop = fir::DoLoopOp::create(builder, loc, zero, len, one);
  auto insPt = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(loop.getBody());
  aiir::Value slen =
      srcLen
          ? fir::ConvertOp::create(builder, loc, one.getType(), srcLen)
                .getResult()
          : aiir::arith::ConstantIndexOp::create(builder, loc, srcTy.getLen())
                .getResult();
  auto cond =
      aiir::arith::CmpIOp::create(builder, loc, aiir::arith::CmpIPredicate::slt,
                                  loop.getInductionVar(), slen);
  auto ifOp = fir::IfOp::create(builder, loc, cond, /*withElse=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto csrcTy = toArrayTy(srcTy);
  auto csrc = fir::ConvertOp::create(builder, loc, csrcTy, src);
  auto in = fir::CoordinateOp::create(builder, loc, toCoorTy(csrcTy), csrc,
                                      loop.getInductionVar());
  auto load = fir::LoadOp::create(builder, loc, in);
  auto cdstTy = toArrayTy(dstTy);
  auto cdst = fir::ConvertOp::create(builder, loc, cdstTy, dst);
  auto out = fir::CoordinateOp::create(builder, loc, toCoorTy(cdstTy), cdst,
                                       loop.getInductionVar());
  aiir::Value cast =
      srcTy.getFKind() == dstTy.getFKind()
          ? load.getResult()
          : fir::ConvertOp::create(builder, loc, toEleTy(cdstTy), load)
                .getResult();
  fir::StoreOp::create(builder, loc, cast, out);
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  auto space = fir::StringLitOp::create(builder, loc, toEleTy(cdstTy),
                                        llvm::ArrayRef<char>{' '});
  auto cdst2 = fir::ConvertOp::create(builder, loc, cdstTy, dst);
  auto out2 = fir::CoordinateOp::create(builder, loc, toCoorTy(cdstTy), cdst2,
                                        loop.getInductionVar());
  fir::StoreOp::create(builder, loc, space, out2);
  builder.restoreInsertionPoint(insPt);
}

/// Get extents from fir.shape/fir.shape_shift op. Empty result if
/// \p shapeVal is empty or is a fir.shift.
inline llvm::SmallVector<aiir::Value> getExtents(aiir::Value shapeVal) {
  if (shapeVal)
    if (auto *shapeOp = shapeVal.getDefiningOp()) {
      if (auto shOp = aiir::dyn_cast<fir::ShapeOp>(shapeOp)) {
        auto operands = shOp.getExtents();
        return {operands.begin(), operands.end()};
      }
      if (auto shOp = aiir::dyn_cast<fir::ShapeShiftOp>(shapeOp)) {
        auto operands = shOp.getExtents();
        return {operands.begin(), operands.end()};
      }
    }
  return {};
}

/// Get origins from fir.shape_shift/fir.shift op. Empty result if
/// \p shapeVal is empty or is a fir.shape.
inline llvm::SmallVector<aiir::Value> getOrigins(aiir::Value shapeVal) {
  if (shapeVal)
    if (auto *shapeOp = shapeVal.getDefiningOp()) {
      if (auto shOp = aiir::dyn_cast<fir::ShapeShiftOp>(shapeOp)) {
        auto operands = shOp.getOrigins();
        return {operands.begin(), operands.end()};
      }
      if (auto shOp = aiir::dyn_cast<fir::ShiftOp>(shapeOp)) {
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
llvm::SmallVector<aiir::Value>
originateIndices(aiir::Location loc, B &builder, aiir::Type memTy,
                 aiir::Value shapeVal, aiir::ValueRange indices) {
  llvm::SmallVector<aiir::Value> result;
  auto origins = getOrigins(shapeVal);
  if (origins.empty()) {
    assert(!shapeVal || aiir::isa<fir::ShapeOp>(shapeVal.getDefiningOp()));
    auto ty = fir::dyn_cast_ptrOrBoxEleTy(memTy);
    assert(ty && aiir::isa<fir::SequenceType>(ty));
    auto seqTy = aiir::cast<fir::SequenceType>(ty);
    auto one = aiir::arith::ConstantIndexOp::create(builder, loc, 1);
    const auto dimension = seqTy.getDimension();
    if (shapeVal) {
      assert(dimension == aiir::cast<fir::ShapeOp>(shapeVal.getDefiningOp())
                              .getType()
                              .getRank());
    }
    for (auto i : llvm::enumerate(indices)) {
      if (i.index() < dimension) {
        assert(fir::isa_integer(i.value().getType()));
        result.push_back(
            aiir::arith::AddIOp::create(builder, loc, i.value(), one));
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
      result.push_back(aiir::arith::AddIOp::create(builder, loc, i.value(),
                                                   origins[origOff++]));
    else
      result.push_back(i.value());
  }
  return result;
}

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_FACTORY_H
