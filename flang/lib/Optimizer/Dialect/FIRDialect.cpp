//===-- FIRDialect.cpp ----------------------------------------------------===//
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

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"

using namespace fir;

fir::FIROpsDialect::FIROpsDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect("fir", ctx, mlir::TypeID::get<FIROpsDialect>()) {
  addTypes<BoxType, BoxCharType, BoxProcType, CharacterType, fir::ComplexType,
           FieldType, HeapType, fir::IntegerType, LenType, LogicalType,
           PointerType, RealType, RecordType, ReferenceType, SequenceType,
           ShapeType, ShapeShiftType, ShiftType, SliceType, TypeDescType,
           fir::VectorType>();
  addAttributes<ClosedIntervalAttr, ExactTypeAttr, LowerBoundAttr,
                PointIntervalAttr, RealAttr, SubclassAttr, UpperBoundAttr>();
  addOperations<
#define GET_OP_LIST
#include "flang/Optimizer/Dialect/FIROps.cpp.inc"
      >();
}

// anchor the class vtable to this compilation unit
fir::FIROpsDialect::~FIROpsDialect() {
  // do nothing
}

mlir::Type fir::FIROpsDialect::parseType(mlir::DialectAsmParser &parser) const {
  return parseFirType(const_cast<FIROpsDialect *>(this), parser);
}

void fir::FIROpsDialect::printType(mlir::Type ty,
                                   mlir::DialectAsmPrinter &p) const {
  return printFirType(const_cast<FIROpsDialect *>(this), ty, p);
}

mlir::Attribute
fir::FIROpsDialect::parseAttribute(mlir::DialectAsmParser &parser,
                                   mlir::Type type) const {
  return parseFirAttribute(const_cast<FIROpsDialect *>(this), parser, type);
}

void fir::FIROpsDialect::printAttribute(mlir::Attribute attr,
                                        mlir::DialectAsmPrinter &p) const {
  printFirAttribute(const_cast<FIROpsDialect *>(this), attr, p);
}
