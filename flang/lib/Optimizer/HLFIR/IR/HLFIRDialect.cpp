//===-- HLFIRDialect.cpp --------------------------------------------------===//
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

#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flang/Optimizer/HLFIR/HLFIRDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flang/Optimizer/HLFIR/HLFIRTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "flang/Optimizer/HLFIR/HLFIRAttributes.cpp.inc"

void hlfir::hlfirDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flang/Optimizer/HLFIR/HLFIRTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "flang/Optimizer/HLFIR/HLFIROps.cpp.inc"
      >();
}

// `expr` `<` `*` | bounds (`x` bounds)* `:` type [`?`] `>`
// bounds ::= `?` | int-lit
mlir::Type hlfir::ExprType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return {};
  ExprType::Shape shape;
  if (parser.parseOptionalStar()) {
    if (parser.parseDimensionList(shape, /*allowDynamic=*/true))
      return {};
  } else if (parser.parseColon()) {
    return {};
  }
  mlir::Type eleTy;
  if (parser.parseType(eleTy))
    return {};
  const bool polymorphic = mlir::succeeded(parser.parseOptionalQuestion());
  if (parser.parseGreater())
    return {};
  return ExprType::get(parser.getContext(), shape, eleTy, polymorphic);
}

void hlfir::ExprType::print(mlir::AsmPrinter &printer) const {
  auto shape = getShape();
  printer << '<';
  if (shape.size()) {
    for (const auto &b : shape) {
      if (b >= 0)
        printer << b << 'x';
      else
        printer << "?x";
    }
  }
  printer << getEleTy();
  if (isPolymorphic())
    printer << '?';
  printer << '>';
}

bool hlfir::isFortranVariableType(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, bool>(type)
      .Case<fir::ReferenceType, fir::PointerType, fir::HeapType>([](auto p) {
        mlir::Type eleType = p.getEleTy();
        return eleType.isa<fir::BaseBoxType>() || !fir::hasDynamicSize(eleType);
      })
      .Case<fir::BaseBoxType, fir::BoxCharType>([](auto) { return true; })
      .Default([](mlir::Type) { return false; });
}

bool hlfir::isFortranScalarCharacterType(mlir::Type type) {
  return isFortranScalarCharacterExprType(type) ||
         type.isa<fir::BoxCharType>() ||
         fir::unwrapPassByRefType(fir::unwrapRefType(type))
             .isa<fir::CharacterType>();
}

bool hlfir::isFortranScalarCharacterExprType(mlir::Type type) {
  if (auto exprType = type.dyn_cast<hlfir::ExprType>())
    return exprType.isScalar() &&
           exprType.getElementType().isa<fir::CharacterType>();
  return false;
}

bool hlfir::isFortranScalarNumericalType(mlir::Type type) {
  return fir::isa_integer(type) || fir::isa_real(type) ||
         fir::isa_complex(type);
}

bool hlfir::isFortranNumericalArrayObject(mlir::Type type) {
  if (isBoxAddressType(type))
    return false;
  if (auto arrayTy =
          getFortranElementOrSequenceType(type).dyn_cast<fir::SequenceType>())
    return isFortranScalarNumericalType(arrayTy.getEleTy());
  return false;
}

bool hlfir::isFortranNumericalOrLogicalArrayObject(mlir::Type type) {
  if (isBoxAddressType(type))
    return false;
  if (auto arrayTy =
          getFortranElementOrSequenceType(type).dyn_cast<fir::SequenceType>()) {
    mlir::Type eleTy = arrayTy.getEleTy();
    return isFortranScalarNumericalType(eleTy) ||
           mlir::isa<fir::LogicalType>(eleTy);
  }
  return false;
}

bool hlfir::isPassByRefOrIntegerType(mlir::Type type) {
  mlir::Type unwrappedType = fir::unwrapPassByRefType(type);
  return fir::isa_integer(unwrappedType);
}

bool hlfir::isI1Type(mlir::Type type) {
  if (mlir::IntegerType integer = type.dyn_cast<mlir::IntegerType>())
    if (integer.getWidth() == 1)
      return true;
  return false;
}

bool hlfir::isMaskArgument(mlir::Type type) {
  if (isBoxAddressType(type))
    return false;

  mlir::Type unwrappedType = fir::unwrapPassByRefType(fir::unwrapRefType(type));
  mlir::Type elementType = getFortranElementType(unwrappedType);
  if (unwrappedType != elementType)
    // input type is an array
    return mlir::isa<fir::LogicalType>(elementType);

  // input is a scalar, so allow i1 too
  return mlir::isa<fir::LogicalType>(elementType) || isI1Type(elementType);
}
