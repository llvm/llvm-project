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

//===-----------------------------------------------------------------===//
// RecordMembers
//===-----------------------------------------------------------------===//

static void printRecordMembers(mlir::AsmPrinter &p, mlir::ArrayAttr members);
static mlir::ParseResult parseRecordMembers(mlir::AsmParser &parser,
                                            mlir::ArrayAttr &members);

//===-----------------------------------------------------------------===//
// IntLiteral
//===-----------------------------------------------------------------===//

static void printIntLiteral(mlir::AsmPrinter &p, llvm::APInt value,
                            cir::IntTypeInterface ty);
static mlir::ParseResult parseIntLiteral(mlir::AsmParser &parser,
                                         llvm::APInt &value,
                                         cir::IntTypeInterface ty);
//===-----------------------------------------------------------------===//
// FloatLiteral
//===-----------------------------------------------------------------===//

static void printFloatLiteral(mlir::AsmPrinter &p, llvm::APFloat value,
                              mlir::Type ty);
static mlir::ParseResult
parseFloatLiteral(mlir::AsmParser &parser,
                  mlir::FailureOr<llvm::APFloat> &value,
                  cir::FPTypeInterface fpType);

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

static void printRecordMembers(mlir::AsmPrinter &printer,
                               mlir::ArrayAttr members) {
  printer << '{';
  llvm::interleaveComma(members, printer);
  printer << '}';
}

static ParseResult parseRecordMembers(mlir::AsmParser &parser,
                                      mlir::ArrayAttr &members) {
  llvm::SmallVector<mlir::Attribute, 4> elts;

  auto delimiter = AsmParser::Delimiter::Braces;
  auto result = parser.parseCommaSeparatedList(delimiter, [&]() {
    mlir::TypedAttr attr;
    if (parser.parseAttribute(attr).failed())
      return mlir::failure();
    elts.push_back(attr);
    return mlir::success();
  });

  if (result.failed())
    return mlir::failure();

  members = mlir::ArrayAttr::get(parser.getContext(), elts);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConstRecordAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult
ConstRecordAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        mlir::Type type, ArrayAttr members) {
  auto sTy = mlir::dyn_cast_if_present<cir::RecordType>(type);
  if (!sTy)
    return emitError() << "expected !cir.record type";

  if (sTy.getMembers().size() != members.size())
    return emitError() << "number of elements must match";

  unsigned attrIdx = 0;
  for (auto &member : sTy.getMembers()) {
    auto m = mlir::cast<mlir::TypedAttr>(members[attrIdx]);
    if (member != m.getType())
      return emitError() << "element at index " << attrIdx << " has type "
                         << m.getType()
                         << " but the expected type for this element is "
                         << member;
    attrIdx++;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// OptInfoAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult OptInfoAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  unsigned level, unsigned size) {
  if (level > 3)
    return emitError()
           << "optimization level must be between 0 and 3 inclusive";
  if (size > 2)
    return emitError()
           << "size optimization level must be between 0 and 2 inclusive";
  return success();
}

//===----------------------------------------------------------------------===//
// ConstPtrAttr definitions
//===----------------------------------------------------------------------===//

// TODO(CIR): Consider encoding the null value differently and use conditional
// assembly format instead of custom parsing/printing.
static ParseResult parseConstPtr(AsmParser &parser, mlir::IntegerAttr &value) {

  if (parser.parseOptionalKeyword("null").succeeded()) {
    value = parser.getBuilder().getI64IntegerAttr(0);
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

template <typename IntT>
static bool isTooLargeForType(const mlir::APInt &value, IntT expectedValue) {
  if constexpr (std::is_signed_v<IntT>) {
    return value.getSExtValue() != expectedValue;
  } else {
    return value.getZExtValue() != expectedValue;
  }
}

template <typename IntT>
static mlir::ParseResult parseIntLiteralImpl(mlir::AsmParser &p,
                                             llvm::APInt &value,
                                             cir::IntTypeInterface ty) {
  IntT ivalue;
  const bool isSigned = ty.isSigned();
  if (p.parseInteger(ivalue))
    return p.emitError(p.getCurrentLocation(), "expected integer value");

  value = mlir::APInt(ty.getWidth(), ivalue, isSigned, /*implicitTrunc=*/true);
  if (isTooLargeForType(value, ivalue))
    return p.emitError(p.getCurrentLocation(),
                       "integer value too large for the given type");

  return success();
}

mlir::ParseResult parseIntLiteral(mlir::AsmParser &parser, llvm::APInt &value,
                                  cir::IntTypeInterface ty) {
  if (ty.isSigned())
    return parseIntLiteralImpl<int64_t>(parser, value, ty);
  return parseIntLiteralImpl<uint64_t>(parser, value, ty);
}

void printIntLiteral(mlir::AsmPrinter &p, llvm::APInt value,
                     cir::IntTypeInterface ty) {
  if (ty.isSigned())
    p << value.getSExtValue();
  else
    p << value.getZExtValue();
}

LogicalResult IntAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              cir::IntTypeInterface type, llvm::APInt value) {
  if (value.getBitWidth() != type.getWidth())
    return emitError() << "type and value bitwidth mismatch: "
                       << type.getWidth() << " != " << value.getBitWidth();
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
                                     cir::FPTypeInterface fpType) {

  APFloat parsedValue(0.0);
  if (parser.parseFloat(fpType.getFloatSemantics(), parsedValue))
    return failure();

  value.emplace(parsedValue);
  return success();
}

FPAttr FPAttr::getZero(Type type) {
  return get(type,
             APFloat::getZero(
                 mlir::cast<cir::FPTypeInterface>(type).getFloatSemantics()));
}

LogicalResult FPAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             cir::FPTypeInterface fpType, APFloat value) {
  if (APFloat::SemanticsToEnum(fpType.getFloatSemantics()) !=
      APFloat::SemanticsToEnum(value.getSemantics()))
    return emitError() << "floating-point semantics mismatch";

  return success();
}

//===----------------------------------------------------------------------===//
// ConstComplexAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult
ConstComplexAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         cir::ComplexType type, mlir::TypedAttr real,
                         mlir::TypedAttr imag) {
  mlir::Type elemType = type.getElementType();
  if (real.getType() != elemType)
    return emitError()
           << "type of the real part does not match the complex type";

  if (imag.getType() != elemType)
    return emitError()
           << "type of the imaginary part does not match the complex type";

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
// CIR VTableAttr
//===----------------------------------------------------------------------===//

LogicalResult cir::VTableAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, mlir::Type type,
    mlir::ArrayAttr data) {
  auto sTy = mlir::dyn_cast_if_present<cir::RecordType>(type);
  if (!sTy)
    return emitError() << "expected !cir.record type result";
  if (sTy.getMembers().empty() || data.empty())
    return emitError() << "expected record type with one or more subtype";

  if (cir::ConstRecordAttr::verify(emitError, type, data).failed())
    return failure();

  for (const auto &element : data.getAsRange<mlir::Attribute>()) {
    const auto &constArrayAttr = mlir::dyn_cast<cir::ConstArrayAttr>(element);
    if (!constArrayAttr)
      return emitError() << "expected constant array subtype";

    LogicalResult eltTypeCheck = success();
    auto arrayElts = mlir::cast<ArrayAttr>(constArrayAttr.getElts());
    arrayElts.walkImmediateSubElements(
        [&](mlir::Attribute attr) {
          if (mlir::isa<ConstPtrAttr, GlobalViewAttr>(attr))
            return;

          eltTypeCheck = emitError()
                         << "expected GlobalViewAttr or ConstPtrAttr";
        },
        [&](mlir::Type type) {});
    if (eltTypeCheck.failed())
      return eltTypeCheck;
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
