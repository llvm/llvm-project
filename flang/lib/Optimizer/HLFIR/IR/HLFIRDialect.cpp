//===-- HLFIRDialect.cpp --------------------------------------------------===//
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

#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/DialectImplementation.h"
#include "aiir/IR/Matchers.h"
#include "aiir/IR/OpImplementation.h"
#include "llvm/ADT/SmallVector.h"
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
aiir::Type hlfir::ExprType::parse(aiir::AsmParser &parser) {
  if (parser.parseLess())
    return {};
  ExprType::Shape shape;
  if (parser.parseOptionalStar()) {
    if (parser.parseDimensionList(shape, /*allowDynamic=*/true))
      return {};
  } else if (parser.parseColon()) {
    return {};
  }
  aiir::Type eleTy;
  if (parser.parseType(eleTy))
    return {};
  const bool polymorphic = aiir::succeeded(parser.parseOptionalQuestion());
  if (parser.parseGreater())
    return {};
  return ExprType::get(parser.getContext(), shape, eleTy, polymorphic);
}

void hlfir::ExprType::print(aiir::AsmPrinter &printer) const {
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

bool hlfir::isFortranVariableType(aiir::Type type) {
  return llvm::TypeSwitch<aiir::Type, bool>(type)
      .Case<fir::ReferenceType, fir::PointerType, fir::HeapType>([](auto p) {
        aiir::Type eleType = p.getEleTy();
        return aiir::isa<fir::BaseBoxType>(eleType) ||
               !fir::hasDynamicSize(eleType);
      })
      .Case<fir::BaseBoxType, fir::BoxCharType>([](aiir::Type) { return true; })
      .Case([](fir::VectorType) { return true; })
      .Default([](aiir::Type) { return false; });
}

bool hlfir::isFortranScalarCharacterType(aiir::Type type) {
  return isFortranScalarCharacterExprType(type) ||
         aiir::isa<fir::BoxCharType>(type) ||
         aiir::isa<fir::CharacterType>(
             fir::unwrapPassByRefType(fir::unwrapRefType(type)));
}

bool hlfir::isFortranScalarCharacterExprType(aiir::Type type) {
  if (auto exprType = aiir::dyn_cast<hlfir::ExprType>(type))
    return exprType.isScalar() &&
           aiir::isa<fir::CharacterType>(exprType.getElementType());
  return false;
}

bool hlfir::isFortranArrayCharacterExprType(aiir::Type type) {
  if (auto exprType = aiir::dyn_cast<hlfir::ExprType>(type))
    return exprType.isArray() &&
           aiir::isa<fir::CharacterType>(exprType.getElementType());

  return false;
}

bool hlfir::isFortranScalarNumericalType(aiir::Type type) {
  return fir::isa_integer(type) || fir::isa_real(type) ||
         fir::isa_complex(type);
}

bool hlfir::isFortranNumericalArrayObject(aiir::Type type) {
  if (isBoxAddressType(type))
    return false;
  if (auto arrayTy = aiir::dyn_cast<fir::SequenceType>(
          getFortranElementOrSequenceType(type)))
    return isFortranScalarNumericalType(arrayTy.getEleTy());
  return false;
}

bool hlfir::isFortranNumericalOrLogicalArrayObject(aiir::Type type) {
  if (isBoxAddressType(type))
    return false;
  if (auto arrayTy = aiir::dyn_cast<fir::SequenceType>(
          getFortranElementOrSequenceType(type))) {
    aiir::Type eleTy = arrayTy.getEleTy();
    return isFortranScalarNumericalType(eleTy) ||
           aiir::isa<fir::LogicalType>(eleTy);
  }
  return false;
}

bool hlfir::isFortranArrayObject(aiir::Type type) {
  if (isBoxAddressType(type))
    return false;
  return !!aiir::dyn_cast<fir::SequenceType>(
      getFortranElementOrSequenceType(type));
}

bool hlfir::isPassByRefOrIntegerType(aiir::Type type) {
  aiir::Type unwrappedType = fir::unwrapPassByRefType(type);
  return fir::isa_integer(unwrappedType);
}

bool hlfir::isI1Type(aiir::Type type) {
  if (aiir::IntegerType integer = aiir::dyn_cast<aiir::IntegerType>(type))
    if (integer.getWidth() == 1)
      return true;
  return false;
}

bool hlfir::isFortranLogicalArrayObject(aiir::Type type) {
  if (isBoxAddressType(type))
    return false;
  if (auto arrayTy = aiir::dyn_cast<fir::SequenceType>(
          getFortranElementOrSequenceType(type))) {
    aiir::Type eleTy = arrayTy.getEleTy();
    return aiir::isa<fir::LogicalType>(eleTy);
  }
  return false;
}

bool hlfir::isMaskArgument(aiir::Type type) {
  if (isBoxAddressType(type))
    return false;

  aiir::Type unwrappedType = fir::unwrapPassByRefType(fir::unwrapRefType(type));
  aiir::Type elementType = getFortranElementType(unwrappedType);
  if (unwrappedType != elementType)
    // input type is an array
    return aiir::isa<fir::LogicalType>(elementType);

  // input is a scalar, so allow i1 too
  return aiir::isa<fir::LogicalType>(elementType) || isI1Type(elementType);
}

bool hlfir::isPolymorphicObject(aiir::Type type) {
  if (auto exprType = aiir::dyn_cast<hlfir::ExprType>(type))
    return exprType.isPolymorphic();

  return fir::isPolymorphicType(type);
}

aiir::Value hlfir::genExprShape(aiir::OpBuilder &builder,
                                const aiir::Location &loc,
                                const hlfir::ExprType &expr) {
  aiir::IndexType indexTy = builder.getIndexType();
  llvm::SmallVector<aiir::Value> extents;
  extents.reserve(expr.getRank());

  for (std::int64_t extent : expr.getShape()) {
    if (extent == hlfir::ExprType::getUnknownExtent())
      return {};
    extents.emplace_back(aiir::arith::ConstantOp::create(
        builder, loc, indexTy, builder.getIntegerAttr(indexTy, extent)));
  }

  fir::ShapeType shapeTy =
      fir::ShapeType::get(builder.getContext(), expr.getRank());
  fir::ShapeOp shape = fir::ShapeOp::create(builder, loc, shapeTy, extents);
  return shape.getResult();
}

bool hlfir::mayHaveAllocatableComponent(aiir::Type ty) {
  return fir::isPolymorphicType(ty) || fir::isUnlimitedPolymorphicType(ty) ||
         fir::isRecordWithAllocatableMember(hlfir::getFortranElementType(ty));
}

aiir::Type hlfir::getExprType(aiir::Type variableType) {
  hlfir::ExprType::Shape typeShape;
  bool isPolymorphic = fir::isPolymorphicType(variableType);
  aiir::Type type = getFortranElementOrSequenceType(variableType);
  if (auto seqType = aiir::dyn_cast<fir::SequenceType>(type)) {
    assert(!seqType.hasUnknownShape() && "assumed-rank cannot be expressions");
    typeShape.append(seqType.getShape().begin(), seqType.getShape().end());
    type = seqType.getEleTy();
  }
  return hlfir::ExprType::get(variableType.getContext(), typeShape, type,
                              isPolymorphic);
}

bool hlfir::isFortranIntegerScalarOrArrayObject(aiir::Type type) {
  if (isBoxAddressType(type))
    return false;

  aiir::Type unwrappedType = fir::unwrapPassByRefType(fir::unwrapRefType(type));
  aiir::Type elementType = getFortranElementType(unwrappedType);
  return aiir::isa<aiir::IntegerType>(elementType);
}
