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
  if (!mlir::isa<IntType>(type)) {
    emitError() << "expected 'simple.int' type";
    return failure();
  }

  auto intType = mlir::cast<IntType>(type);
  if (value.getBitWidth() != intType.getWidth()) {
    emitError() << "type and value bitwidth mismatch: " << intType.getWidth()
                << " != " << value.getBitWidth();
    return failure();
  }

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
      APFloat::SemanticsToEnum(value.getSemantics())) {
    emitError() << "floating-point semantics mismatch";
    return failure();
  }

  return success();
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
