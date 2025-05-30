//===- CIRAttrs.cpp - MLIR CIR Attributes ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the attributes in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

static void printFloatLiteral(mlir::AsmPrinter &p, llvm::APFloat value,
                              mlir::Type ty);
static mlir::ParseResult
parseFloatLiteral(mlir::AsmParser &parser,
                  mlir::FailureOr<llvm::APFloat> &value,
                  cir::CIRFPTypeInterface fpType);

static mlir::ParseResult parseConstPtr(mlir::AsmParser &parser,
                                       mlir::IntegerAttr &value);

static void printConstPtr(mlir::AsmPrinter &p, mlir::IntegerAttr value);

#define GET_ATTRDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsAttributes.cpp.inc"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// General CIR parsing / printing
//===----------------------------------------------------------------------===//

Attribute CIRDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  llvm::StringRef mnemonic;
  Attribute genAttr;
  OptionalParseResult parseResult =
      generatedAttributeParser(parser, &mnemonic, type, genAttr);
  if (parseResult.has_value())
    return genAttr;
  parser.emitError(typeLoc, "unknown attribute in CIR dialect");
  return Attribute();
}

void CIRDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  if (failed(generatedAttributePrinter(attr, os)))
    llvm_unreachable("unexpected CIR type kind");
}

//===----------------------------------------------------------------------===//
// ConstPtrAttr definitions
//===----------------------------------------------------------------------===//

// TODO(CIR): Consider encoding the null value differently and use conditional
// assembly format instead of custom parsing/printing.
static ParseResult parseConstPtr(AsmParser &parser, mlir::IntegerAttr &value) {

  if (parser.parseOptionalKeyword("null").succeeded()) {
    value = mlir::IntegerAttr::get(
        mlir::IntegerType::get(parser.getContext(), 64), 0);
    return success();
  }

  return parser.parseAttribute(value);
}

static void printConstPtr(AsmPrinter &p, mlir::IntegerAttr value) {
  if (!value.getInt())
    p << "null";
  else
    p << value;
}

//===----------------------------------------------------------------------===//
// IntAttr definitions
//===----------------------------------------------------------------------===//

Attribute IntAttr::parse(AsmParser &parser, Type odsType) {
  mlir::APInt apValue;

  if (!mlir::isa<IntType>(odsType))
    return {};
  auto type = mlir::cast<IntType>(odsType);

  // Consume the '<' symbol.
  if (parser.parseLess())
    return {};

  // Fetch arbitrary precision integer value.
  if (type.isSigned()) {
    int64_t value = 0;
    if (parser.parseInteger(value)) {
      parser.emitError(parser.getCurrentLocation(), "expected integer value");
    } else {
      apValue = mlir::APInt(type.getWidth(), value, type.isSigned(),
                            /*implicitTrunc=*/true);
      if (apValue.getSExtValue() != value)
        parser.emitError(parser.getCurrentLocation(),
                         "integer value too large for the given type");
    }
  } else {
    uint64_t value = 0;
    if (parser.parseInteger(value)) {
      parser.emitError(parser.getCurrentLocation(), "expected integer value");
    } else {
      apValue = mlir::APInt(type.getWidth(), value, type.isSigned(),
                            /*implicitTrunc=*/true);
      if (apValue.getZExtValue() != value)
        parser.emitError(parser.getCurrentLocation(),
                         "integer value too large for the given type");
    }
  }

  // Consume the '>' symbol.
  if (parser.parseGreater())
    return {};

  return IntAttr::get(type, apValue);
}

void IntAttr::print(AsmPrinter &printer) const {
  auto type = mlir::cast<IntType>(getType());
  printer << '<';
  if (type.isSigned())
    printer << getSInt();
  else
    printer << getUInt();
  printer << '>';
}

LogicalResult IntAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              Type type, APInt value) {
  if (!mlir::isa<IntType>(type))
    return emitError() << "expected 'simple.int' type";

  auto intType = mlir::cast<IntType>(type);
  if (value.getBitWidth() != intType.getWidth())
    return emitError() << "type and value bitwidth mismatch: "
                       << intType.getWidth() << " != " << value.getBitWidth();

  return success();
}

//===----------------------------------------------------------------------===//
// FPAttr definitions
//===----------------------------------------------------------------------===//

static void printFloatLiteral(AsmPrinter &p, APFloat value, Type ty) {
  p << value;
}

static ParseResult parseFloatLiteral(AsmParser &parser,
                                     FailureOr<APFloat> &value,
                                     CIRFPTypeInterface fpType) {

  APFloat parsedValue(0.0);
  if (parser.parseFloat(fpType.getFloatSemantics(), parsedValue))
    return failure();

  value.emplace(parsedValue);
  return success();
}

FPAttr FPAttr::getZero(Type type) {
  return get(type,
             APFloat::getZero(
                 mlir::cast<CIRFPTypeInterface>(type).getFloatSemantics()));
}

LogicalResult FPAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             CIRFPTypeInterface fpType, APFloat value) {
  if (APFloat::SemanticsToEnum(fpType.getFloatSemantics()) !=
      APFloat::SemanticsToEnum(value.getSemantics()))
    return emitError() << "floating-point semantics mismatch";

  return success();
}

//===----------------------------------------------------------------------===//
// CIR ConstArrayAttr
//===----------------------------------------------------------------------===//

LogicalResult
ConstArrayAttr::verify(function_ref<InFlightDiagnostic()> emitError, Type type,
                       Attribute elts, int trailingZerosNum) {

  if (!(mlir::isa<ArrayAttr, StringAttr>(elts)))
    return emitError() << "constant array expects ArrayAttr or StringAttr";

  if (auto strAttr = mlir::dyn_cast<StringAttr>(elts)) {
    const auto arrayTy = mlir::cast<ArrayType>(type);
    const auto intTy = mlir::dyn_cast<IntType>(arrayTy.getElementType());

    // TODO: add CIR type for char.
    if (!intTy || intTy.getWidth() != 8)
      return emitError()
             << "constant array element for string literals expects "
                "!cir.int<u, 8> element type";
    return success();
  }

  assert(mlir::isa<ArrayAttr>(elts));
  const auto arrayAttr = mlir::cast<mlir::ArrayAttr>(elts);
  const auto arrayTy = mlir::cast<ArrayType>(type);

  // Make sure both number of elements and subelement types match type.
  if (arrayTy.getSize() != arrayAttr.size() + trailingZerosNum)
    return emitError() << "constant array size should match type size";
  return success();
}

Attribute ConstArrayAttr::parse(AsmParser &parser, Type type) {
  mlir::FailureOr<Type> resultTy;
  mlir::FailureOr<Attribute> resultVal;

  // Parse literal '<'
  if (parser.parseLess())
    return {};

  // Parse variable 'value'
  resultVal = FieldParser<Attribute>::parse(parser);
  if (failed(resultVal)) {
    parser.emitError(
        parser.getCurrentLocation(),
        "failed to parse ConstArrayAttr parameter 'value' which is "
        "to be a `Attribute`");
    return {};
  }

  // ArrayAttrrs have per-element type, not the type of the array...
  if (mlir::isa<ArrayAttr>(*resultVal)) {
    // Array has implicit type: infer from const array type.
    if (parser.parseOptionalColon().failed()) {
      resultTy = type;
    } else { // Array has explicit type: parse it.
      resultTy = FieldParser<Type>::parse(parser);
      if (failed(resultTy)) {
        parser.emitError(
            parser.getCurrentLocation(),
            "failed to parse ConstArrayAttr parameter 'type' which is "
            "to be a `::mlir::Type`");
        return {};
      }
    }
  } else {
    auto ta = mlir::cast<TypedAttr>(*resultVal);
    resultTy = ta.getType();
    if (mlir::isa<mlir::NoneType>(*resultTy)) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected type declaration for string literal");
      return {};
    }
  }

  unsigned zeros = 0;
  if (parser.parseOptionalComma().succeeded()) {
    if (parser.parseOptionalKeyword("trailing_zeros").succeeded()) {
      unsigned typeSize =
          mlir::cast<cir::ArrayType>(resultTy.value()).getSize();
      mlir::Attribute elts = resultVal.value();
      if (auto str = mlir::dyn_cast<mlir::StringAttr>(elts))
        zeros = typeSize - str.size();
      else
        zeros = typeSize - mlir::cast<mlir::ArrayAttr>(elts).size();
    } else {
      return {};
    }
  }

  // Parse literal '>'
  if (parser.parseGreater())
    return {};

  return parser.getChecked<ConstArrayAttr>(
      parser.getCurrentLocation(), parser.getContext(), resultTy.value(),
      resultVal.value(), zeros);
}

void ConstArrayAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getElts());
  if (getTrailingZerosNum())
    printer << ", trailing_zeros";
  printer << ">";
}

//===----------------------------------------------------------------------===//
// CIR ConstVectorAttr
//===----------------------------------------------------------------------===//

LogicalResult
cir::ConstVectorAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             Type type, ArrayAttr elts) {

  if (!mlir::isa<cir::VectorType>(type))
    return emitError() << "type of cir::ConstVectorAttr is not a "
                          "cir::VectorType: "
                       << type;

  const auto vecType = mlir::cast<cir::VectorType>(type);

  if (vecType.getSize() != elts.size())
    return emitError()
           << "number of constant elements should match vector size";

  // Check if the types of the elements match
  LogicalResult elementTypeCheck = success();
  elts.walkImmediateSubElements(
      [&](Attribute element) {
        if (elementTypeCheck.failed()) {
          // An earlier element didn't match
          return;
        }
        auto typedElement = mlir::dyn_cast<TypedAttr>(element);
        if (!typedElement ||
            typedElement.getType() != vecType.getElementType()) {
          elementTypeCheck = failure();
          emitError() << "constant type should match vector element type";
        }
      },
      [&](Type) {});

  return elementTypeCheck;
}

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//

void CIRDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "clang/CIR/Dialect/IR/CIROpsAttributes.cpp.inc"
      >();
}
