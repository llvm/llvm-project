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

/// Get extents from fir.shape/fir.shape_shift op. Empty result if
/// \p shapeVal is empty or is a fir.shift.
inline std::vector<mlir::Value> getExtents(mlir::Value shapeVal) {
  if (shapeVal)
    if (auto *shapeOp = shapeVal.getDefiningOp()) {
      if (auto shOp = mlir::dyn_cast<fir::ShapeOp>(shapeOp)) {
        auto operands = shOp.getExtents();
        return {operands.begin(), operands.end()};
      }
      if (auto shOp = mlir::dyn_cast<fir::ShapeShiftOp>(shapeOp))
        return shOp.getExtents();
    }
  return {};
}

/// Get origins from fir.shape_shift/fir.shift op. Empty result if
/// \p shapeVal is empty or is a fir.shape.
inline std::vector<mlir::Value> getOrigins(mlir::Value shapeVal) {
  if (shapeVal)
    if (auto *shapeOp = shapeVal.getDefiningOp()) {
      if (auto shOp = mlir::dyn_cast<fir::ShapeShiftOp>(shapeOp))
        return shOp.getOrigins();
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
    const auto dimension = seqTy.getDimension();
    assert(shapeVal &&
           dimension == mlir::cast<fir::ShapeOp>(shapeVal.getDefiningOp())
                            .getType()
                            .getRank());
    auto one = builder.template create<arith::ConstantIndexOp>(loc, 1);
    for (auto i : llvm::enumerate(indices)) {
      if (i.index() < dimension) {
        assert(fir::isa_integer(i.value().getType()));
        result.push_back(
            builder.template create<arith::AddIOp>(loc, i.value(), one));
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
      result.push_back(builder.template create<arith::AddIOp>(
          loc, i.value(), origins[origOff++]));
    else
      result.push_back(i.value());
  }
  return result;
}

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_FACTORY_H
