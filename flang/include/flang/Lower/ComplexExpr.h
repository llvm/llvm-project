//===-- Lower/ComplexExpr.h -- lowering of complex values -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_COMPLEXEXPR_H
#define FORTRAN_LOWER_COMPLEXEXPR_H

#include "flang/Lower/FIRBuilder.h"

namespace Fortran::lower {

/// Helper to facilitate lowering of COMPLEX manipulations in FIR.
class ComplexExprHelper {
public:
  explicit ComplexExprHelper(FirOpBuilder &builder, mlir::Location loc)
      : builder(builder), loc(loc) {}
  ComplexExprHelper(const ComplexExprHelper &) = delete;

  // The values of part enum members are meaningful for
  // InsertValueOp and ExtractValueOp so they are explicit.
  enum class Part { Real = 0, Imag = 1 };

  /// Type helper. Determine the type. Do not create MLIR operations.
  mlir::Type getComplexPartType(mlir::Value cplx);
  mlir::Type getComplexPartType(mlir::Type complexType);

  /// Complex operation creation helper. They create MLIR operations.
  mlir::Value createComplex(fir::KindTy kind, mlir::Value real,
                            mlir::Value imag);

  /// Create a complex value.
  mlir::Value createComplex(mlir::Type complexType, mlir::Value real,
                            mlir::Value imag);

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

  mlir::Value createComplexCompare(mlir::Value cplx1, mlir::Value cplx2,
                                   bool eq);

protected:
  template <Part partId>
  mlir::Value extract(mlir::Value cplx) {
    return builder.create<fir::ExtractValueOp>(loc, getComplexPartType(cplx),
                                               cplx, createPartId<partId>());
  }

  template <Part partId>
  mlir::Value insert(mlir::Value cplx, mlir::Value part) {
    return builder.create<fir::InsertValueOp>(loc, cplx.getType(), cplx, part,
                                              createPartId<partId>());
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

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_COMPLEXEXPR_H
