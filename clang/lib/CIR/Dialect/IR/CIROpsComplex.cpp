//===- CIROpsComplex.cpp - CIR ops in partition `Complex` -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR ops in partition `Complex`: the hand-written
// parts first, then the TableGen-generated definitions.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIROpsComplex.h"

#include "CIROpDefinitions.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// ComplexCreateOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ComplexCreateOp::verify() {
  if (getType().getElementType() != getReal().getType()) {
    emitOpError()
        << "operand type of cir.complex.create does not match its result type";
    return failure();
  }

  return success();
}

OpFoldResult cir::ComplexCreateOp::fold(FoldAdaptor adaptor) {
  mlir::Attribute real = adaptor.getReal();
  mlir::Attribute imag = adaptor.getImag();
  if (!real || !imag)
    return {};

  // When both of real and imag are constants, we can fold the operation into an
  // `#cir.const_complex` operation.
  auto realAttr = mlir::cast<mlir::TypedAttr>(real);
  auto imagAttr = mlir::cast<mlir::TypedAttr>(imag);
  return cir::ConstComplexAttr::get(realAttr, imagAttr);
}

//===----------------------------------------------------------------------===//
// ComplexRealOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ComplexRealOp::verify() {
  mlir::Type operandTy = getOperand().getType();
  if (auto complexOperandTy = mlir::dyn_cast<cir::ComplexType>(operandTy))
    operandTy = complexOperandTy.getElementType();

  if (getType() != operandTy) {
    emitOpError() << ": result type does not match operand type";
    return failure();
  }

  return success();
}

OpFoldResult cir::ComplexRealOp::fold(FoldAdaptor adaptor) {
  if (!mlir::isa<cir::ComplexType>(getOperand().getType()))
    return nullptr;

  if (auto complexCreateOp = getOperand().getDefiningOp<cir::ComplexCreateOp>())
    return complexCreateOp.getOperand(0);

  auto complex =
      mlir::cast_if_present<cir::ConstComplexAttr>(adaptor.getOperand());
  return complex ? complex.getReal() : nullptr;
}

//===----------------------------------------------------------------------===//
// ComplexImagOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ComplexImagOp::verify() {
  mlir::Type operandTy = getOperand().getType();
  if (auto complexOperandTy = mlir::dyn_cast<cir::ComplexType>(operandTy))
    operandTy = complexOperandTy.getElementType();

  if (getType() != operandTy) {
    emitOpError() << ": result type does not match operand type";
    return failure();
  }

  return success();
}

OpFoldResult cir::ComplexImagOp::fold(FoldAdaptor adaptor) {
  if (!mlir::isa<cir::ComplexType>(getOperand().getType()))
    return nullptr;

  if (auto complexCreateOp = getOperand().getDefiningOp<cir::ComplexCreateOp>())
    return complexCreateOp.getOperand(1);

  auto complex =
      mlir::cast_if_present<cir::ConstComplexAttr>(adaptor.getOperand());
  return complex ? complex.getImag() : nullptr;
}

//===----------------------------------------------------------------------===//
// ComplexRealPtrOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ComplexRealPtrOp::verify() {
  mlir::Type resultPointeeTy = getType().getPointee();
  cir::PointerType operandPtrTy = getOperand().getType();
  auto operandPointeeTy =
      mlir::cast<cir::ComplexType>(operandPtrTy.getPointee());

  if (resultPointeeTy != operandPointeeTy.getElementType()) {
    return emitOpError() << ": result type does not match operand type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ComplexImagPtrOp
//===----------------------------------------------------------------------===//

LogicalResult cir::ComplexImagPtrOp::verify() {
  mlir::Type resultPointeeTy = getType().getPointee();
  cir::PointerType operandPtrTy = getOperand().getType();
  auto operandPointeeTy =
      mlir::cast<cir::ComplexType>(operandPtrTy.getPointee());

  if (resultPointeeTy != operandPointeeTy.getElementType()) {
    return emitOpError()
           << "cir.complex.imag_ptr result type does not match operand type";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsComplex.cpp.inc"
