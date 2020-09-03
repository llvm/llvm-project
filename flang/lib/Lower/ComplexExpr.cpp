//===-- ComplexExpr.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ComplexExpr.h"
#include "flang/Lower/ConvertType.h"

//===----------------------------------------------------------------------===//
// ComplexExprHelper implementation
//===----------------------------------------------------------------------===//

mlir::Type
Fortran::lower::ComplexExprHelper::getComplexPartType(mlir::Type complexType) {
  return Fortran::lower::convertReal(
      builder.getContext(), complexType.cast<fir::CplxType>().getFKind());
}

mlir::Type
Fortran::lower::ComplexExprHelper::getComplexPartType(mlir::Value cplx) {
  return getComplexPartType(cplx.getType());
}

mlir::Value Fortran::lower::ComplexExprHelper::createComplex(fir::KindTy kind,
                                                             mlir::Value real,
                                                             mlir::Value imag) {
  auto complexTy = fir::CplxType::get(builder.getContext(), kind);
  mlir::Value und = builder.create<fir::UndefOp>(loc, complexTy);
  return insert<Part::Imag>(insert<Part::Real>(und, real), imag);
}

mlir::Value Fortran::lower::ComplexExprHelper::createComplex(mlir::Type cplxTy,
                                                             mlir::Value real,
                                                             mlir::Value imag) {
  mlir::Value und = builder.create<fir::UndefOp>(loc, cplxTy);
  return insert<Part::Imag>(insert<Part::Real>(und, real), imag);
}

mlir::Value Fortran::lower::ComplexExprHelper::createComplexCompare(
    mlir::Value cplx1, mlir::Value cplx2, bool eq) {
  auto real1 = extract<Part::Real>(cplx1);
  auto real2 = extract<Part::Real>(cplx2);
  auto imag1 = extract<Part::Imag>(cplx1);
  auto imag2 = extract<Part::Imag>(cplx2);

  mlir::CmpFPredicate predicate =
      eq ? mlir::CmpFPredicate::UEQ : mlir::CmpFPredicate::UNE;
  mlir::Value realCmp =
      builder.create<mlir::CmpFOp>(loc, predicate, real1, real2);
  mlir::Value imagCmp =
      builder.create<mlir::CmpFOp>(loc, predicate, imag1, imag2);

  return eq ? builder.create<mlir::AndOp>(loc, realCmp, imagCmp).getResult()
            : builder.create<mlir::OrOp>(loc, realCmp, imagCmp).getResult();
}
