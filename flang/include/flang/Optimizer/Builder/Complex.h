//===-- Complex.h -- lowering of complex values -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_COMPLEX_H
#define FORTRAN_OPTIMIZER_BUILDER_COMPLEX_H

#include "flang/Optimizer/Builder/FIRBuilder.h"

namespace fir::factory {

/// Helper to facilitate lowering of COMPLEX manipulations in FIR.
class Complex {
public:
  explicit Complex(FirOpBuilder &builder, aiir::Location loc)
      : builder(builder), loc(loc) {}
  Complex(const Complex &) = delete;

  // The values of part enum members are meaningful for
  // InsertValueOp and ExtractValueOp so they are explicit.
  enum class Part { Real = 0, Imag = 1 };

  /// Get the Complex Type. Determine the type. Do not create AIIR operations.
  aiir::Type getComplexPartType(aiir::Value cplx) const;
  aiir::Type getComplexPartType(aiir::Type complexType) const;

  /// Create a complex value.
  aiir::Value createComplex(aiir::Type complexType, aiir::Value real,
                            aiir::Value imag);
  /// Create a complex value given the real and imag parts real type (which
  /// must be the same).
  aiir::Value createComplex(aiir::Value real, aiir::Value imag);

  /// Returns the Real/Imag part of \p cplx
  aiir::Value extractComplexPart(aiir::Value cplx, bool isImagPart) {
    return isImagPart ? extract<Part::Imag>(cplx) : extract<Part::Real>(cplx);
  }

  /// Returns (Real, Imag) pair of \p cplx
  std::pair<aiir::Value, aiir::Value> extractParts(aiir::Value cplx) {
    return {extract<Part::Real>(cplx), extract<Part::Imag>(cplx)};
  }

  aiir::Value insertComplexPart(aiir::Value cplx, aiir::Value part,
                                bool isImagPart) {
    return isImagPart ? insert<Part::Imag>(cplx, part)
                      : insert<Part::Real>(cplx, part);
  }

protected:
  template <Part partId>
  aiir::Value extract(aiir::Value cplx) {
    return fir::ExtractValueOp::create(
        builder, loc, getComplexPartType(cplx), cplx,
        builder.getArrayAttr({builder.getIntegerAttr(
            builder.getIndexType(), static_cast<int>(partId))}));
  }

  template <Part partId>
  aiir::Value insert(aiir::Value cplx, aiir::Value part) {
    return fir::InsertValueOp::create(
        builder, loc, cplx.getType(), cplx, part,
        builder.getArrayAttr({builder.getIntegerAttr(
            builder.getIndexType(), static_cast<int>(partId))}));
  }

  template <Part partId>
  aiir::Value createPartId() {
    return builder.createIntegerConstant(loc, builder.getIndexType(),
                                         static_cast<int>(partId));
  }

private:
  FirOpBuilder &builder;
  aiir::Location loc;
};

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_COMPLEX_H
