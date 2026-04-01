//===- CIRAttrs.cpp - AIIR CIR Attributes ---------------------------------===//
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

#include "aiir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "aiir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

//===-----------------------------------------------------------------===//
// RecordMembers
//===-----------------------------------------------------------------===//

static void printRecordMembers(aiir::AsmPrinter &p, aiir::ArrayAttr members);
static aiir::ParseResult parseRecordMembers(aiir::AsmParser &parser,
                                            aiir::ArrayAttr &members);

//===-----------------------------------------------------------------===//
// IntLiteral
//===-----------------------------------------------------------------===//

static void printIntLiteral(aiir::AsmPrinter &p, llvm::APInt value,
                            cir::IntTypeInterface ty);
static aiir::ParseResult parseIntLiteral(aiir::AsmParser &parser,
                                         llvm::APInt &value,
                                         cir::IntTypeInterface ty);
//===-----------------------------------------------------------------===//
// FloatLiteral
//===-----------------------------------------------------------------===//

static void printFloatLiteral(aiir::AsmPrinter &p, llvm::APFloat value,
                              aiir::Type ty);
static aiir::ParseResult
parseFloatLiteral(aiir::AsmParser &parser,
                  aiir::FailureOr<llvm::APFloat> &value,
                  cir::FPTypeInterface fpType);

//===----------------------------------------------------------------------===//
// AddressSpaceAttr
//===----------------------------------------------------------------------===//

aiir::ParseResult parseAddressSpaceValue(aiir::AsmParser &p,
                                         cir::LangAddressSpace &addrSpace) {
  llvm::SMLoc loc = p.getCurrentLocation();
  aiir::FailureOr<cir::LangAddressSpace> result =
      aiir::FieldParser<cir::LangAddressSpace>::parse(p);
  if (aiir::failed(result))
    return p.emitError(loc, "expected address space keyword");
  addrSpace = result.value();
  return aiir::success();
}

void printAddressSpaceValue(aiir::AsmPrinter &p,
                            cir::LangAddressSpace addrSpace) {
  p << cir::stringifyEnum(addrSpace);
}

static aiir::ParseResult parseConstPtr(aiir::AsmParser &parser,
                                       aiir::IntegerAttr &value);

static void printConstPtr(aiir::AsmPrinter &p, aiir::IntegerAttr value);

#define GET_ATTRDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsAttributes.cpp.inc"

using namespace aiir;
using namespace cir;

//===----------------------------------------------------------------------===//
// MemorySpaceAttrInterface implementations for Lang and Target address space
// attributes
//===----------------------------------------------------------------------===//

bool LangAddressSpaceAttr::isValidLoad(
    aiir::Type type, aiir::ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const aiir::DataLayout *dataLayout,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidLoad for LangAddressSpaceAttr NYI");
}

bool LangAddressSpaceAttr::isValidStore(
    aiir::Type type, aiir::ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const aiir::DataLayout *dataLayout,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidStore for LangAddressSpaceAttr NYI");
}

bool LangAddressSpaceAttr::isValidAtomicOp(
    aiir::ptr::AtomicBinOp op, aiir::Type type,
    aiir::ptr::AtomicOrdering ordering, std::optional<int64_t> alignment,
    const aiir::DataLayout *dataLayout,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidAtomicOp for LangAddressSpaceAttr NYI");
}

bool LangAddressSpaceAttr::isValidAtomicXchg(
    aiir::Type type, aiir::ptr::AtomicOrdering successOrdering,
    aiir::ptr::AtomicOrdering failureOrdering, std::optional<int64_t> alignment,
    const aiir::DataLayout *dataLayout,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidAtomicXchg for LangAddressSpaceAttr NYI");
}

bool LangAddressSpaceAttr::isValidAddrSpaceCast(
    aiir::Type tgt, aiir::Type src,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidAddrSpaceCast for LangAddressSpaceAttr NYI");
}

bool LangAddressSpaceAttr::isValidPtrIntCast(
    aiir::Type intLikeTy, aiir::Type ptrLikeTy,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidPtrIntCast for LangAddressSpaceAttr NYI");
}

bool TargetAddressSpaceAttr::isValidLoad(
    aiir::Type type, aiir::ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const aiir::DataLayout *dataLayout,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidLoad for TargetAddressSpaceAttr NYI");
}

bool TargetAddressSpaceAttr::isValidStore(
    aiir::Type type, aiir::ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const aiir::DataLayout *dataLayout,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidStore for TargetAddressSpaceAttr NYI");
}

bool TargetAddressSpaceAttr::isValidAtomicOp(
    aiir::ptr::AtomicBinOp op, aiir::Type type,
    aiir::ptr::AtomicOrdering ordering, std::optional<int64_t> alignment,
    const aiir::DataLayout *dataLayout,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidAtomicOp for TargetAddressSpaceAttr NYI");
}

bool TargetAddressSpaceAttr::isValidAtomicXchg(
    aiir::Type type, aiir::ptr::AtomicOrdering successOrdering,
    aiir::ptr::AtomicOrdering failureOrdering, std::optional<int64_t> alignment,
    const aiir::DataLayout *dataLayout,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidAtomicXchg for TargetAddressSpaceAttr NYI");
}

bool TargetAddressSpaceAttr::isValidAddrSpaceCast(
    aiir::Type tgt, aiir::Type src,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidAddrSpaceCast for TargetAddressSpaceAttr NYI");
}

bool TargetAddressSpaceAttr::isValidPtrIntCast(
    aiir::Type intLikeTy, aiir::Type ptrLikeTy,
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError) const {
  llvm_unreachable("isValidPtrIntCast for TargetAddressSpaceAttr NYI");
}

//===----------------------------------------------------------------------===//
// General CIR parsing / printing
//===----------------------------------------------------------------------===//

static void printRecordMembers(aiir::AsmPrinter &printer,
                               aiir::ArrayAttr members) {
  printer << '{';
  llvm::interleaveComma(members, printer);
  printer << '}';
}

static ParseResult parseRecordMembers(aiir::AsmParser &parser,
                                      aiir::ArrayAttr &members) {
  llvm::SmallVector<aiir::Attribute, 4> elts;

  auto delimiter = AsmParser::Delimiter::Braces;
  auto result = parser.parseCommaSeparatedList(delimiter, [&]() {
    aiir::TypedAttr attr;
    if (parser.parseAttribute(attr).failed())
      return aiir::failure();
    elts.push_back(attr);
    return aiir::success();
  });

  if (result.failed())
    return aiir::failure();

  members = aiir::ArrayAttr::get(parser.getContext(), elts);
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// ConstRecordAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult
ConstRecordAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        aiir::Type type, ArrayAttr members) {
  auto sTy = aiir::dyn_cast_if_present<cir::RecordType>(type);
  if (!sTy)
    return emitError() << "expected !cir.record type";

  if (sTy.getMembers().size() != members.size())
    return emitError() << "number of elements must match";

  unsigned attrIdx = 0;
  for (auto &member : sTy.getMembers()) {
    auto m = aiir::cast<aiir::TypedAttr>(members[attrIdx]);
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
static ParseResult parseConstPtr(AsmParser &parser, aiir::IntegerAttr &value) {

  if (parser.parseOptionalKeyword("null").succeeded()) {
    value = parser.getBuilder().getI64IntegerAttr(0);
    return success();
  }

  return parser.parseAttribute(value);
}

static void printConstPtr(AsmPrinter &p, aiir::IntegerAttr value) {
  if (!value.getInt())
    p << "null";
  else
    p << value;
}

//===----------------------------------------------------------------------===//
// IntAttr definitions
//===----------------------------------------------------------------------===//

template <typename IntT>
static bool isTooLargeForType(const aiir::APInt &value, IntT expectedValue) {
  if constexpr (std::is_signed_v<IntT>) {
    return value.getSExtValue() != expectedValue;
  } else {
    return value.getZExtValue() != expectedValue;
  }
}

template <typename IntT>
static aiir::ParseResult parseIntLiteralImpl(aiir::AsmParser &p,
                                             llvm::APInt &value,
                                             cir::IntTypeInterface ty) {
  IntT ivalue;
  const bool isSigned = ty.isSigned();
  if (p.parseInteger(ivalue))
    return p.emitError(p.getCurrentLocation(), "expected integer value");

  value = aiir::APInt(ty.getWidth(), ivalue, isSigned, /*implicitTrunc=*/true);
  if (isTooLargeForType(value, ivalue))
    return p.emitError(p.getCurrentLocation(),
                       "integer value too large for the given type");

  return success();
}

aiir::ParseResult parseIntLiteral(aiir::AsmParser &parser, llvm::APInt &value,
                                  cir::IntTypeInterface ty) {
  if (ty.isSigned())
    return parseIntLiteralImpl<int64_t>(parser, value, ty);
  return parseIntLiteralImpl<uint64_t>(parser, value, ty);
}

void printIntLiteral(aiir::AsmPrinter &p, llvm::APInt value,
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
                 aiir::cast<cir::FPTypeInterface>(type).getFloatSemantics()));
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
                         cir::ComplexType type, aiir::TypedAttr real,
                         aiir::TypedAttr imag) {
  aiir::Type elemType = type.getElementType();
  if (real.getType() != elemType)
    return emitError()
           << "type of the real part does not match the complex type";

  if (imag.getType() != elemType)
    return emitError()
           << "type of the imaginary part does not match the complex type";

  return success();
}

//===----------------------------------------------------------------------===//
// DataMemberAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult
DataMemberAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       cir::DataMemberType ty,
                       std::optional<unsigned> memberIndex) {
  // DataMemberAttr without a given index represents a null value.
  if (!memberIndex.has_value())
    return success();

  cir::RecordType recTy = ty.getClassTy();
  if (recTy.isIncomplete())
    return emitError()
           << "incomplete 'cir.record' cannot be used to build a non-null "
              "data member pointer";

  unsigned memberIndexValue = memberIndex.value();
  if (memberIndexValue >= recTy.getNumElements())
    return emitError()
           << "member index of a #cir.data_member attribute is out of range";

  aiir::Type memberTy = recTy.getMembers()[memberIndexValue];
  if (memberTy != ty.getMemberTy())
    return emitError()
           << "member type of a #cir.data_member attribute must match the "
              "attribute type";

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
  auto ty = aiir::cast<cir::MethodType>(odsType);

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
  aiir::OptionalParseResult parseSymbolRefResult =
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

  if (!(aiir::isa<ArrayAttr, StringAttr>(elts)))
    return emitError() << "constant array expects ArrayAttr or StringAttr";

  if (auto strAttr = aiir::dyn_cast<StringAttr>(elts)) {
    const auto arrayTy = aiir::cast<ArrayType>(type);
    const auto intTy = aiir::dyn_cast<IntType>(arrayTy.getElementType());

    // TODO: add CIR type for char.
    if (!intTy || intTy.getWidth() != 8)
      return emitError()
             << "constant array element for string literals expects "
                "!cir.int<u, 8> element type";
    return success();
  }

  assert(aiir::isa<ArrayAttr>(elts));
  const auto arrayAttr = aiir::cast<aiir::ArrayAttr>(elts);
  const auto arrayTy = aiir::cast<ArrayType>(type);

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
  aiir::FailureOr<Type> resultTy;
  aiir::FailureOr<Attribute> resultVal;

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
  if (aiir::isa<ArrayAttr>(*resultVal)) {
    // Array has implicit type: infer from const array type.
    if (parser.parseOptionalColon().failed()) {
      resultTy = type;
    } else { // Array has explicit type: parse it.
      resultTy = FieldParser<Type>::parse(parser);
      if (failed(resultTy)) {
        parser.emitError(
            parser.getCurrentLocation(),
            "failed to parse ConstArrayAttr parameter 'type' which is "
            "to be a `::aiir::Type`");
        return {};
      }
    }
  } else {
    auto ta = aiir::cast<TypedAttr>(*resultVal);
    resultTy = ta.getType();
    if (aiir::isa<aiir::NoneType>(*resultTy)) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected type declaration for string literal");
      return {};
    }
  }

  unsigned zeros = 0;
  if (parser.parseOptionalComma().succeeded()) {
    if (parser.parseOptionalKeyword("trailing_zeros").succeeded()) {
      unsigned typeSize =
          aiir::cast<cir::ArrayType>(resultTy.value()).getSize();
      aiir::Attribute elts = resultVal.value();
      if (auto str = aiir::dyn_cast<aiir::StringAttr>(elts))
        zeros = typeSize - str.size();
      else
        zeros = typeSize - aiir::cast<aiir::ArrayAttr>(elts).size();
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

  if (!aiir::isa<cir::VectorType>(type))
    return emitError() << "type of cir::ConstVectorAttr is not a "
                          "cir::VectorType: "
                       << type;

  const auto vecType = aiir::cast<cir::VectorType>(type);

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
        auto typedElement = aiir::dyn_cast<TypedAttr>(element);
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
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError, aiir::Type type,
    aiir::ArrayAttr data) {
  auto sTy = aiir::dyn_cast_if_present<cir::RecordType>(type);
  if (!sTy)
    return emitError() << "expected !cir.record type result";
  if (sTy.getMembers().empty() || data.empty())
    return emitError() << "expected record type with one or more subtype";

  if (cir::ConstRecordAttr::verify(emitError, type, data).failed())
    return failure();

  for (const auto &element : data.getAsRange<aiir::Attribute>()) {
    const auto &constArrayAttr = aiir::dyn_cast<cir::ConstArrayAttr>(element);
    if (!constArrayAttr)
      return emitError() << "expected constant array subtype";

    LogicalResult eltTypeCheck = success();
    auto arrayElts = aiir::cast<ArrayAttr>(constArrayAttr.getElts());
    arrayElts.walkImmediateSubElements(
        [&](aiir::Attribute attr) {
          if (aiir::isa<ConstPtrAttr, GlobalViewAttr>(attr))
            return;

          eltTypeCheck = emitError()
                         << "expected GlobalViewAttr or ConstPtrAttr";
        },
        [&](aiir::Type type) {});
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
    cir::GlobalViewAttr destRtti, aiir::FlatSymbolRefAttr runtimeFunc,
    aiir::FlatSymbolRefAttr badCastFunc, cir::IntAttr offsetHint) {
  auto isRttiPtr = [](aiir::Type ty) {
    // RTTI pointers are !cir.ptr<!u8i>.

    auto ptrTy = aiir::dyn_cast<cir::PointerType>(ty);
    if (!ptrTy)
      return false;

    auto pointeeIntTy = aiir::dyn_cast<cir::IntType>(ptrTy.getPointee());
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
// CIR Dialect
//===----------------------------------------------------------------------===//

void CIRDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "clang/CIR/Dialect/IR/CIROpsAttributes.cpp.inc"
      >();
}
