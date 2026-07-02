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

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "mlir/IR/Attributes.h"
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

//===----------------------------------------------------------------------===//
// AddressSpaceAttr
//===----------------------------------------------------------------------===//

mlir::ParseResult parseAddressSpaceValue(mlir::AsmParser &p,
                                         cir::LangAddressSpace &addrSpace) {
  llvm::SMLoc loc = p.getCurrentLocation();
  mlir::FailureOr<cir::LangAddressSpace> result =
      mlir::FieldParser<cir::LangAddressSpace>::parse(p);
  if (mlir::failed(result))
    return p.emitError(loc, "expected address space keyword");
  addrSpace = result.value();
  return mlir::success();
}

void printAddressSpaceValue(mlir::AsmPrinter &p,
                            cir::LangAddressSpace addrSpace) {
  p << cir::stringifyEnum(addrSpace);
}

static mlir::ParseResult parseConstPtr(mlir::AsmParser &parser,
                                       mlir::IntegerAttr &value);

static void printConstPtr(mlir::AsmPrinter &p, mlir::IntegerAttr value);

static mlir::ParseResult
parseDataMemberPath(mlir::AsmParser &parser,
                    mlir::DenseI32ArrayAttr &memberPath);

static void printDataMemberPath(mlir::AsmPrinter &p,
                                mlir::DenseI32ArrayAttr memberPath);

#define GET_ATTRDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsAttributes.cpp.inc"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// MemorySpaceAttrInterface implementations for Lang and Target address space
// attributes
//===----------------------------------------------------------------------===//

bool LangAddressSpaceAttr::isValidLoad(
    mlir::Type type, mlir::ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const mlir::DataLayout *dataLayout,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidLoad for LangAddressSpaceAttr NYI");
}

bool LangAddressSpaceAttr::isValidStore(
    mlir::Type type, mlir::ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const mlir::DataLayout *dataLayout,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidStore for LangAddressSpaceAttr NYI");
}

bool LangAddressSpaceAttr::isValidAtomicOp(
    mlir::ptr::AtomicBinOp op, mlir::Type type,
    mlir::ptr::AtomicOrdering ordering, std::optional<int64_t> alignment,
    const mlir::DataLayout *dataLayout,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidAtomicOp for LangAddressSpaceAttr NYI");
}

bool LangAddressSpaceAttr::isValidAtomicXchg(
    mlir::Type type, mlir::ptr::AtomicOrdering successOrdering,
    mlir::ptr::AtomicOrdering failureOrdering, std::optional<int64_t> alignment,
    const mlir::DataLayout *dataLayout,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidAtomicXchg for LangAddressSpaceAttr NYI");
}

bool LangAddressSpaceAttr::isValidAddrSpaceCast(
    mlir::Type tgt, mlir::Type src,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidAddrSpaceCast for LangAddressSpaceAttr NYI");
}

bool LangAddressSpaceAttr::isValidPtrIntCast(
    mlir::Type intLikeTy, mlir::Type ptrLikeTy,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidPtrIntCast for LangAddressSpaceAttr NYI");
}

bool TargetAddressSpaceAttr::isValidLoad(
    mlir::Type type, mlir::ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const mlir::DataLayout *dataLayout,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidLoad for TargetAddressSpaceAttr NYI");
}

bool TargetAddressSpaceAttr::isValidStore(
    mlir::Type type, mlir::ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const mlir::DataLayout *dataLayout,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidStore for TargetAddressSpaceAttr NYI");
}

bool TargetAddressSpaceAttr::isValidAtomicOp(
    mlir::ptr::AtomicBinOp op, mlir::Type type,
    mlir::ptr::AtomicOrdering ordering, std::optional<int64_t> alignment,
    const mlir::DataLayout *dataLayout,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidAtomicOp for TargetAddressSpaceAttr NYI");
}

bool TargetAddressSpaceAttr::isValidAtomicXchg(
    mlir::Type type, mlir::ptr::AtomicOrdering successOrdering,
    mlir::ptr::AtomicOrdering failureOrdering, std::optional<int64_t> alignment,
    const mlir::DataLayout *dataLayout,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidAtomicXchg for TargetAddressSpaceAttr NYI");
}

bool TargetAddressSpaceAttr::isValidAddrSpaceCast(
    mlir::Type tgt, mlir::Type src,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidAddrSpaceCast for TargetAddressSpaceAttr NYI");
}

bool TargetAddressSpaceAttr::isValidPtrIntCast(
    mlir::Type intLikeTy, mlir::Type ptrLikeTy,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidPtrIntCast for TargetAddressSpaceAttr NYI");
}

//===----------------------------------------------------------------------===//
// General CIR parsing / printing
//===----------------------------------------------------------------------===//

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
    return emitError() << "expected !cir.struct or !cir.union type";

  // union record initializer is just a single element that has to match one of
  // the fields in the union (the new active member).
  if (sTy.isUnion()) {
    if (members.size() != 1)
      return emitError() << "union constant must have exactly one element, got "
                         << members.size();
    auto m = mlir::cast<mlir::TypedAttr>(members[0]);
    if (!llvm::is_contained(sTy.getMembers(), m.getType()))
      return emitError() << "union element type " << m.getType()
                         << " is not a member of " << sTy;
    return success();
  }

  if (sTy.getMembers().size() != members.size())
    return emitError() << "number of elements must match";

  for (const auto &[attrIdx, member] : llvm::enumerate(sTy.getMembers())) {
    auto m = mlir::cast<mlir::TypedAttr>(members[attrIdx]);

    // As a special case, we allow a flexible array member. This can only be the
    // last element, the rest of the array type has to match (that is, the
    // element type has to match), and the array member must be size zero.
    if (attrIdx == sTy.getMembers().size() - 1) {
      auto memArrayTy = dyn_cast<cir::ArrayType>(member);
      if (memArrayTy && memArrayTy.getSize() == 0) {

        // The FAM must only match another array type initializer.
        if (!isa<cir::ArrayType>(m.getType()))
          return emitError()
                 << "element at index " << attrIdx << " has type "
                 << m.getType() << " but the expected type for this element is "
                 << member;

        cir::ArrayType initArrayTy = cast<cir::ArrayType>(m.getType());
        // The FAM only matches an equivalent array type.
        if (initArrayTy.getElementType() != memArrayTy.getElementType())
          return emitError()
                 << "flexible array member at index " << attrIdx << " has type "
                 << m.getType()
                 << " which doesn't match the expected element type of member "
                 << member;
        continue;
      }
    }

    if (member != m.getType())
      return emitError() << "element at index " << attrIdx << " has type "
                         << m.getType()
                         << " but the expected type for this element is "
                         << member;
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

static ParseResult parseDataMemberPath(AsmParser &parser,
                                       mlir::DenseI32ArrayAttr &memberPath) {
  if (parser.parseOptionalKeyword("null").succeeded())
    return success();

  auto parsed = mlir::FieldParser<mlir::DenseI32ArrayAttr>::parse(parser);
  if (mlir::failed(parsed))
    return failure();
  memberPath = *parsed;
  return success();
}

static void printDataMemberPath(AsmPrinter &p,
                                mlir::DenseI32ArrayAttr memberPath) {
  if (!memberPath)
    p << "null";
  else
    p.printStrippedAttrOrType(memberPath);
}

//===----------------------------------------------------------------------===//
// IntAttr definitions
//===----------------------------------------------------------------------===//

mlir::ParseResult parseIntLiteral(mlir::AsmParser &parser, llvm::APInt &value,
                                  cir::IntTypeInterface ty) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  llvm::APInt parsed;
  mlir::OptionalParseResult result = parser.parseOptionalInteger(parsed);
  if (!result.has_value() || failed(*result))
    return parser.emitError(loc, "expected integer value");

  const unsigned width = ty.getWidth();
  const bool fits =
      ty.isSigned() ? parsed.getSignificantBits() <= width
                    : !parsed.isNegative() && parsed.getActiveBits() <= width;
  if (!fits)
    return parser.emitError(loc, "integer value too large for the given type");

  value = ty.isSigned() ? parsed.sextOrTrunc(width) : parsed.zextOrTrunc(width);
  return success();
}

void printIntLiteral(mlir::AsmPrinter &p, llvm::APInt value,
                     cir::IntTypeInterface ty) {
  llvm::SmallString<40> str;
  value.toString(str, /*radix=*/10, /*isSigned=*/ty.isSigned());
  p << str;
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
// CmpThreeWayInfoAttr definitions
//===----------------------------------------------------------------------===//

std::string CmpThreeWayInfoAttr::getAlias() const {
  std::string alias = "cmpinfo";

  switch (getOrdering()) {
  case CmpOrdering::Strong:
    alias.append("_strong_");
    break;
  case CmpOrdering::Weak:
    alias.append("_weak_");
    break;
  case CmpOrdering::Partial:
    alias.append("_partial_");
    break;
  }

  auto appendInt = [&](int64_t value) {
    if (value < 0) {
      alias.push_back('n');
      value = -value;
    }
    alias.append(std::to_string(value));
  };

  alias.append("lt");
  appendInt(getLt());
  alias.append("eq");
  appendInt(getEq());
  alias.append("gt");
  appendInt(getGt());

  if (std::optional<int> unordered = getUnordered()) {
    alias.append("un");
    appendInt(unordered.value());
  }

  return alias;
}

LogicalResult
CmpThreeWayInfoAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            CmpOrdering ordering, int64_t lt, int64_t eq,
                            int64_t gt, std::optional<int64_t> unordered) {
  // The presence of unordered must match the value of ordering.
  if ((ordering == CmpOrdering::Strong || ordering == CmpOrdering::Weak) &&
      unordered) {
    emitError() << "strong and weak ordering do not include unordered";
    return failure();
  }
  if (ordering == CmpOrdering::Partial && !unordered) {
    emitError() << "partial ordering requires unordered value";
    return failure();
  }

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
// CIR_CUDAVarRegistrationInfoAttr definitions
//===----------------------------------------------------------------------===//

void CUDAVarRegistrationInfoAttr::print(AsmPrinter &p) const {
  p << "<" << getDeviceSideName();
  p << ", " << stringifyEnum(getKind());
  if (getIsExtern())
    p << ", extern";
  if (getIsConstant())
    p << ", constant";
  if (getIsManaged())
    p << ", managed";
  p << ">";
}

Attribute CUDAVarRegistrationInfoAttr::parse(AsmParser &parser, Type odsType) {
  if (parser.parseLess())
    return {};

  std::string deviceSideName;
  if (parser.parseKeywordOrString(&deviceSideName)) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected device variable name");
    return {};
  }

  if (parser.parseComma())
    return {};

  // Parse the device variable kind (Variable, Surface, Texture)
  StringRef kindStr;
  if (parser.parseKeyword(&kindStr))
    return {};

  std::optional<CUDADeviceVarKind> kind = symbolizeCUDADeviceVarKind(kindStr);
  if (!kind) {
    parser.emitError(parser.getCurrentLocation(),
                     "unknown device variable kind: ")
        << kindStr;
    return {};
  }

  // Parse optional flags: extern, constant, managed
  bool isExtern = false;
  bool isConstant = false;
  bool isManaged = false;

  while (parser.parseOptionalGreater().failed()) {
    if (parser.parseComma())
      return {};

    StringRef flag;
    if (parser.parseKeyword(&flag))
      return {};

    if (flag == "extern")
      isExtern = true;
    else if (flag == "constant")
      isConstant = true;
    else if (flag == "managed")
      isManaged = true;
    else {
      parser.emitError(parser.getCurrentLocation(), "unknown flag: ") << flag;
      return {};
    }
  }

  return get(parser.getContext(), deviceSideName, *kind, isExtern, isConstant,
             isManaged);
}

//===----------------------------------------------------------------------===//
// DataMemberAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult
DataMemberAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       cir::DataMemberType ty,
                       mlir::DenseI32ArrayAttr memberPath) {
  if (!memberPath)
    return success(); // null pointer — always valid

  if (memberPath.empty())
    return emitError() << "#cir.data_member path must not be empty";

  mlir::Type currentTy = ty.getClassTy();
  for (auto [step, idx] : llvm::enumerate(memberPath.asArrayRef())) {
    auto recTy = mlir::dyn_cast<cir::RecordType>(currentTy);
    if (!recTy)
      return emitError() << "#cir.data_member path step " << step
                         << " reaches a non-record type";

    if (recTy.isIncomplete())
      return success(); // cannot validate further; trust the builder

    if (idx < 0 || static_cast<unsigned>(idx) >= recTy.getNumElements())
      return emitError() << "#cir.data_member path index " << idx << " at step "
                         << step << " is out of range";

    currentTy = recTy.getMembers()[idx];
  }

  if (currentTy != ty.getMemberTy())
    return emitError()
           << "member type of a #cir.data_member attribute must match "
              "the attribute type";

  return success();
}

//===----------------------------------------------------------------------===//
// MethodAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult MethodAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 cir::MethodType type,
                                 std::optional<FlatSymbolRefAttr> symbol,
                                 std::optional<uint64_t> vtable_offset) {
  if (symbol.has_value() && vtable_offset.has_value())
    return emitError()
           << "at most one of symbol and vtable_offset can be present "
              "in #cir.method";

  return success();
}

Attribute MethodAttr::parse(AsmParser &parser, Type odsType) {
  auto ty = mlir::cast<cir::MethodType>(odsType);

  if (parser.parseLess().failed())
    return {};

  // Try to parse the null pointer constant.
  if (parser.parseOptionalKeyword("null").succeeded()) {
    if (parser.parseGreater().failed())
      return {};
    return get(ty);
  }

  // Try to parse a flat symbol ref for a pointer to non-virtual member
  // function.
  FlatSymbolRefAttr symbol;
  mlir::OptionalParseResult parseSymbolRefResult =
      parser.parseOptionalAttribute(symbol);
  if (parseSymbolRefResult.has_value()) {
    if (parseSymbolRefResult.value().failed())
      return {};
    if (parser.parseGreater().failed())
      return {};
    return get(ty, symbol);
  }

  // Parse a uint64 that represents the vtable offset.
  std::uint64_t vtableOffset = 0;
  if (parser.parseKeyword("vtable_offset"))
    return {};
  if (parser.parseEqual())
    return {};
  if (parser.parseInteger(vtableOffset))
    return {};

  if (parser.parseGreater())
    return {};

  return get(ty, vtableOffset);
}

void MethodAttr::print(AsmPrinter &printer) const {
  auto symbol = getSymbol();
  auto vtableOffset = getVtableOffset();

  printer << '<';
  if (symbol.has_value()) {
    printer << *symbol;
  } else if (vtableOffset.has_value()) {
    printer << "vtable_offset = " << *vtableOffset;
  } else {
    printer << "null";
  }
  printer << '>';
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
  if (arrayAttr.size() > arrayTy.getSize())
    return emitError() << "constant array has " << arrayAttr.size()
                       << " values but array type has size "
                       << arrayTy.getSize();
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
      unsigned totalSize = mlir::cast<cir::ArrayType>(type).getSize();
      mlir::Attribute elts = resultVal.value();
      if (auto str = mlir::dyn_cast<mlir::StringAttr>(elts))
        zeros = totalSize - str.size();
      else
        zeros = totalSize - mlir::cast<mlir::ArrayAttr>(elts).size();
    } else {
      return {};
    }
  }

  // Parse literal '>'
  if (parser.parseGreater())
    return {};

  return parser.getChecked<ConstArrayAttr>(parser.getCurrentLocation(),
                                           parser.getContext(), type,
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
    return emitError() << "expected !cir.struct or !cir.union type result";
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
// DynamicCastInfoAtttr definitions
//===----------------------------------------------------------------------===//

std::string DynamicCastInfoAttr::getAlias() const {
  // The alias looks like: `dyn_cast_info_<src>_<dest>`

  std::string alias = "dyn_cast_info_";

  alias.append(getSrcRtti().getSymbol().getValue());
  alias.push_back('_');
  alias.append(getDestRtti().getSymbol().getValue());

  return alias;
}

LogicalResult DynamicCastInfoAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, cir::GlobalViewAttr srcRtti,
    cir::GlobalViewAttr destRtti, mlir::FlatSymbolRefAttr runtimeFunc,
    mlir::FlatSymbolRefAttr badCastFunc, cir::IntAttr offsetHint) {
  auto isRttiPtr = [](mlir::Type ty) {
    // RTTI pointers are !cir.ptr<!u8i>.

    auto ptrTy = mlir::dyn_cast<cir::PointerType>(ty);
    if (!ptrTy)
      return false;

    auto pointeeIntTy = mlir::dyn_cast<cir::IntType>(ptrTy.getPointee());
    if (!pointeeIntTy)
      return false;

    return pointeeIntTy.isUnsigned() && pointeeIntTy.getWidth() == 8;
  };

  if (!isRttiPtr(srcRtti.getType()))
    return emitError() << "srcRtti must be an RTTI pointer";

  if (!isRttiPtr(destRtti.getType()))
    return emitError() << "destRtti must be an RTTI pointer";

  return success();
}

//===----------------------------------------------------------------------===//
// RecordLayout lookup
//===----------------------------------------------------------------------===//

RecordLayoutAttr cir::getRecordLayout(mlir::ModuleOp module,
                                      mlir::StringAttr name) {
  auto dict = module->getAttrOfType<mlir::DictionaryAttr>(
      CIRDialect::getRecordLayoutsAttrName());
  assert(dict && "module missing cir.record_layouts attribute");
  auto attr = dict.getAs<RecordLayoutAttr>(name);
  assert(attr && "record layout entry missing for named record");
  return attr;
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
