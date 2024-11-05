//===-- Complex.h -- lowering of complex values -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_COMPLEX_H
#define FORTRAN_OPTIMIZER_BUILDER_COMPLEX_H

#include "flang/Optimizer/Builder/FIRBuilder.h"

namespace fir::factory {

/// Helper to facilitate lowering of COMPLEX manipulations in FIR.
class Complex {
public:
  explicit Complex(FirOpBuilder &builder, mlir::Location loc)
      : builder(builder), loc(loc) {}
  Complex(const Complex &) = delete;

  // The values of part enum members are meaningful for
  // InsertValueOp and ExtractValueOp so they are explicit.
  enum class Part { Real = 0, Imag = 1 };

  /// Get the Complex Type. Determine the type. Do not create MLIR operations.
  mlir::Type getComplexPartType(mlir::Value cplx) const;
  mlir::Type getComplexPartType(mlir::Type complexType) const;

  /// Create a complex value.
  mlir::Value createComplex(mlir::Type complexType, mlir::Value real,
                            mlir::Value imag);
  /// Create a complex value given the real and imag parts real type (which
  /// must be the same).
  mlir::Value createComplex(mlir::Value real, mlir::Value imag);

  /// Returns the Real/Imag part of \p cplx
  mlir::Value extractComplexPart(mlir::Value cplx, bool isImagPart) {
    return isImagPart ? extract<Part::Imag>(cplx) : extract<Part::Real>(cplx);
  }

  /// Returns (Real, Imag) pair of \p cplx
  std::pair<mlir::Value, mlir::Value> extractParts(mlir::Value cplx) {
    return {extract<Part::Real>(cplx), extract<Part::Imag>(cplx)};
  }

  mlir::Value insertComplexPart(mlir::Value cplx, mlir::Value part,
                                bool isImagPart) {
    return isImagPart ? insert<Part::Imag>(cplx, part)
                      : insert<Part::Real>(cplx, part);
  }

protected:
  template <Part partId>
  mlir::Value extract(mlir::Value cplx) {
    return builder.create<fir::ExtractValueOp>(
        loc, getComplexPartType(cplx), cplx,
        builder.getArrayAttr({builder.getIntegerAttr(
            builder.getIndexType(), static_cast<int>(partId))}));
  }

  template <Part partId>
  mlir::Value insert(mlir::Value cplx, mlir::Value part) {
    return builder.create<fir::InsertValueOp>(
        loc, cplx.getType(), cplx, part,
        builder.getArrayAttr({builder.getIntegerAttr(
            builder.getIndexType(), static_cast<int>(partId))}));
  }

  template <Part partId>
  mlir::Value createPartId() {
    return builder.createIntegerConstant(loc, builder.getIndexType(),
                                         static_cast<int>(partId));
  }

private:
  FirOpBuilder &builder;
  mlir::Location loc;
};

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_COMPLEX_H
