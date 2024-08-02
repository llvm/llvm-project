//===- CIRTypes.cpp - MLIR CIR Types --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

// ClangIR holds back AST references when available.
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"

static void printStructMembers(mlir::AsmPrinter &p, mlir::ArrayAttr members);
static mlir::ParseResult parseStructMembers(::mlir::AsmParser &parser,
                                            mlir::ArrayAttr &members);

static void printFloatLiteral(mlir::AsmPrinter &p, llvm::APFloat value,
                              mlir::Type ty);
static mlir::ParseResult
parseFloatLiteral(mlir::AsmParser &parser,
                  mlir::FailureOr<llvm::APFloat> &value, mlir::Type ty);

static mlir::ParseResult parseConstPtr(mlir::AsmParser &parser,
                                       mlir::IntegerAttr &value);

static void printConstPtr(mlir::AsmPrinter &p, mlir::IntegerAttr value);

#define GET_ATTRDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsAttributes.cpp.inc"

using namespace mlir;
using namespace mlir::cir;

//===----------------------------------------------------------------------===//
// CIR AST Attr helpers
//===----------------------------------------------------------------------===//

namespace mlir {
namespace cir {

mlir::Attribute makeFuncDeclAttr(const clang::Decl *decl,
                                 mlir::MLIRContext *ctx) {
  return llvm::TypeSwitch<const clang::Decl *, mlir::Attribute>(decl)
      .Case([ctx](const clang::CXXConstructorDecl *ast) {
        return ASTCXXConstructorDeclAttr::get(ctx, ast);
      })
      .Case([ctx](const clang::CXXConversionDecl *ast) {
        return ASTCXXConversionDeclAttr::get(ctx, ast);
      })
      .Case([ctx](const clang::CXXDestructorDecl *ast) {
        return ASTCXXDestructorDeclAttr::get(ctx, ast);
      })
      .Case([ctx](const clang::CXXMethodDecl *ast) {
        return ASTCXXMethodDeclAttr::get(ctx, ast);
      })
      .Case([ctx](const clang::FunctionDecl *ast) {
        return ASTFunctionDeclAttr::get(ctx, ast);
      })
      .Default([](auto) {
        llvm_unreachable("unexpected Decl kind");
        return mlir::Attribute();
      });
}

} // namespace cir
} // namespace mlir

//===----------------------------------------------------------------------===//
// General CIR parsing / printing
//===----------------------------------------------------------------------===//

Attribute CIRDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef mnemonic;
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

static void printStructMembers(mlir::AsmPrinter &printer,
                               mlir::ArrayAttr members) {
  printer << '{';
  llvm::interleaveComma(members, printer);
  printer << '}';
}

static ParseResult parseStructMembers(mlir::AsmParser &parser,
                                      mlir::ArrayAttr &members) {
  SmallVector<mlir::Attribute, 4> elts;

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

LogicalResult ConstStructAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::Type type, ArrayAttr members) {
  auto sTy = mlir::dyn_cast_if_present<mlir::cir::StructType>(type);
  if (!sTy) {
    emitError() << "expected !cir.struct type";
    return failure();
  }

  if (sTy.getMembers().size() != members.size()) {
    emitError() << "number of elements must match";
    return failure();
  }

  unsigned attrIdx = 0;
  for (auto &member : sTy.getMembers()) {
    auto m = dyn_cast_if_present<TypedAttr>(members[attrIdx]);
    if (!m) {
      emitError() << "expected mlir::TypedAttr attribute";
      return failure();
    }
    if (member != m.getType()) {
      emitError() << "element at index " << attrIdx << " has type "
                  << m.getType() << " but return type for this element is "
                  << member;
      return failure();
    }
    attrIdx++;
  }

  return success();
}

LogicalResult StructLayoutAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned size,
    unsigned alignment, bool padded, mlir::Type largest_member,
    mlir::ArrayAttr offsets) {
  if (not std::all_of(offsets.begin(), offsets.end(), [](mlir::Attribute attr) {
        return mlir::isa<mlir::IntegerAttr>(attr);
      })) {
    return emitError() << "all index values must be integers";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LangAttr definitions
//===----------------------------------------------------------------------===//

Attribute LangAttr::parse(AsmParser &parser, Type odsType) {
  auto loc = parser.getCurrentLocation();
  if (parser.parseLess())
    return {};

  // Parse variable 'lang'.
  llvm::StringRef lang;
  if (parser.parseKeyword(&lang))
    return {};

  // Check if parsed value is a valid language.
  auto langEnum = symbolizeSourceLanguage(lang);
  if (!langEnum.has_value()) {
    parser.emitError(loc) << "invalid language keyword '" << lang << "'";
    return {};
  }

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), langEnum.value());
}

void LangAttr::print(AsmPrinter &printer) const {
  printer << "<" << getLang() << '>';
}

//===----------------------------------------------------------------------===//
// ConstPtrAttr definitions
//===----------------------------------------------------------------------===//

// TODO: Consider encoding the null value differently and use conditional
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
  mlir::APInt APValue;

  if (!mlir::isa<IntType>(odsType))
    return {};
  auto type = mlir::cast<IntType>(odsType);

  // Consume the '<' symbol.
  if (parser.parseLess())
    return {};

  // Fetch arbitrary precision integer value.
  if (type.isSigned()) {
    int64_t value;
    if (parser.parseInteger(value))
      parser.emitError(parser.getCurrentLocation(), "expected integer value");
    APValue = mlir::APInt(type.getWidth(), value, type.isSigned());
    if (APValue.getSExtValue() != value)
      parser.emitError(parser.getCurrentLocation(),
                       "integer value too large for the given type");
  } else {
    uint64_t value;
    if (parser.parseInteger(value))
      parser.emitError(parser.getCurrentLocation(), "expected integer value");
    APValue = mlir::APInt(type.getWidth(), value, type.isSigned());
    if (APValue.getZExtValue() != value)
      parser.emitError(parser.getCurrentLocation(),
                       "integer value too large for the given type");
  }

  // Consume the '>' symbol.
  if (parser.parseGreater())
    return {};

  return IntAttr::get(type, APValue);
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

static void printFloatLiteral(mlir::AsmPrinter &p, llvm::APFloat value,
                              mlir::Type ty) {
  p << value;
}

static mlir::ParseResult
parseFloatLiteral(mlir::AsmParser &parser,
                  mlir::FailureOr<llvm::APFloat> &value, mlir::Type ty) {
  double rawValue;
  if (parser.parseFloat(rawValue)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected floating-point value");
  }

  auto losesInfo = false;
  value.emplace(rawValue);

  auto tyFpInterface = dyn_cast<cir::CIRFPTypeInterface>(ty);
  if (!tyFpInterface) {
    // Parsing of the current floating-point literal has succeeded, but the
    // given attribute type is invalid. This error will be reported later when
    // the attribute is being verified.
    return success();
  }

  value->convert(tyFpInterface.getFloatSemantics(),
                 llvm::RoundingMode::TowardZero, &losesInfo);
  return success();
}

cir::FPAttr cir::FPAttr::getZero(mlir::Type type) {
  return get(
      type, APFloat::getZero(
                mlir::cast<cir::CIRFPTypeInterface>(type).getFloatSemantics()));
}

LogicalResult cir::FPAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type type, APFloat value) {
  auto fltTypeInterface = mlir::dyn_cast<cir::CIRFPTypeInterface>(type);
  if (!fltTypeInterface) {
    emitError() << "expected floating-point type";
    return failure();
  }
  if (APFloat::SemanticsToEnum(fltTypeInterface.getFloatSemantics()) !=
      APFloat::SemanticsToEnum(value.getSemantics())) {
    emitError() << "floating-point semantics mismatch";
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CmpThreeWayInfoAttr definitions
//===----------------------------------------------------------------------===//

std::string CmpThreeWayInfoAttr::getAlias() const {
  std::string alias = "cmp3way_info";

  if (getOrdering() == CmpOrdering::Strong)
    alias.append("_strong_");
  else
    alias.append("_partial_");

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

  if (auto unordered = getUnordered()) {
    alias.append("un");
    appendInt(unordered.value());
  }

  return alias;
}

LogicalResult
CmpThreeWayInfoAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            CmpOrdering ordering, int64_t lt, int64_t eq,
                            int64_t gt, std::optional<int64_t> unordered) {
  // The presense of unordered must match the value of ordering.
  if (ordering == CmpOrdering::Strong && unordered) {
    emitError() << "strong ordering does not include unordered ordering";
    return failure();
  }
  if (ordering == CmpOrdering::Partial && !unordered) {
    emitError() << "partial ordering lacks unordered ordering";
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DataMemberAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult
DataMemberAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       mlir::cir::DataMemberType ty,
                       std::optional<unsigned> memberIndex) {
  if (!memberIndex.has_value()) {
    // DataMemberAttr without a given index represents a null value.
    return success();
  }

  auto clsStructTy = ty.getClsTy();
  if (clsStructTy.isIncomplete()) {
    emitError() << "incomplete 'cir.struct' cannot be used to build a non-null "
                   "data member pointer";
    return failure();
  }

  auto memberIndexValue = memberIndex.value();
  if (memberIndexValue >= clsStructTy.getNumElements()) {
    emitError()
        << "member index of a #cir.data_member attribute is out of range";
    return failure();
  }

  auto memberTy = clsStructTy.getMembers()[memberIndexValue];
  if (memberTy != ty.getMemberTy()) {
    emitError() << "member type of a #cir.data_member attribute must match the "
                   "attribute type";
    return failure();
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
    function_ref<InFlightDiagnostic()> emitError,
    mlir::cir::GlobalViewAttr srcRtti, mlir::cir::GlobalViewAttr destRtti,
    mlir::FlatSymbolRefAttr runtimeFunc, mlir::FlatSymbolRefAttr badCastFunc,
    mlir::cir::IntAttr offsetHint) {
  auto isRttiPtr = [](mlir::Type ty) {
    // RTTI pointers are !cir.ptr<!u8i>.

    auto ptrTy = mlir::dyn_cast<mlir::cir::PointerType>(ty);
    if (!ptrTy)
      return false;

    auto pointeeIntTy = mlir::dyn_cast<mlir::cir::IntType>(ptrTy.getPointee());
    if (!pointeeIntTy)
      return false;

    return pointeeIntTy.isUnsigned() && pointeeIntTy.getWidth() == 8;
  };

  if (!isRttiPtr(srcRtti.getType())) {
    emitError() << "srcRtti must be an RTTI pointer";
    return failure();
  }

  if (!isRttiPtr(destRtti.getType())) {
    emitError() << "destRtti must be an RTTI pointer";
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// OpenCLKernelMetadataAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult OpenCLKernelMetadataAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ArrayAttr workGroupSizeHint, ArrayAttr reqdWorkGroupSize,
    TypeAttr vecTypeHint, std::optional<bool> vecTypeHintSignedness,
    IntegerAttr intelReqdSubGroupSize) {
  // If no field is present, the attribute is considered invalid.
  if (!workGroupSizeHint && !reqdWorkGroupSize && !vecTypeHint &&
      !vecTypeHintSignedness && !intelReqdSubGroupSize) {
    return emitError()
           << "metadata attribute without any field present is invalid";
  }

  // Check for 3-dim integer tuples
  auto is3dimIntTuple = [](ArrayAttr arr) {
    auto isInt = [](Attribute dim) { return mlir::isa<IntegerAttr>(dim); };
    return arr.size() == 3 && llvm::all_of(arr, isInt);
  };
  if (workGroupSizeHint && !is3dimIntTuple(workGroupSizeHint)) {
    return emitError()
           << "work_group_size_hint must have exactly 3 integer elements";
  }
  if (reqdWorkGroupSize && !is3dimIntTuple(reqdWorkGroupSize)) {
    return emitError()
           << "reqd_work_group_size must have exactly 3 integer elements";
  }

  // Check for co-presence of vecTypeHintSignedness
  if (!!vecTypeHint != vecTypeHintSignedness.has_value()) {
    return emitError() << "vec_type_hint_signedness should be present if and "
                          "only if vec_type_hint is set";
  }

  if (vecTypeHint) {
    Type vecTypeHintValue = vecTypeHint.getValue();
    if (mlir::isa<cir::CIRDialect>(vecTypeHintValue.getDialect())) {
      // Check for signedness alignment in CIR
      if (isSignedHint(vecTypeHintValue) != vecTypeHintSignedness) {
        return emitError() << "vec_type_hint_signedness must match the "
                              "signedness of the vec_type_hint type";
      }
      // Check for the dialect of type hint
    } else if (!LLVM::isCompatibleType(vecTypeHintValue)) {
      return emitError() << "vec_type_hint must be a type from the CIR or LLVM "
                            "dialect";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// OpenCLKernelArgMetadataAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult OpenCLKernelArgMetadataAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ArrayAttr addrSpaces, ArrayAttr accessQuals, ArrayAttr types,
    ArrayAttr baseTypes, ArrayAttr typeQuals, ArrayAttr argNames) {
  auto isIntArray = [](ArrayAttr elt) {
    return llvm::all_of(
        elt, [](Attribute elt) { return mlir::isa<IntegerAttr>(elt); });
  };
  auto isStrArray = [](ArrayAttr elt) {
    return llvm::all_of(
        elt, [](Attribute elt) { return mlir::isa<StringAttr>(elt); });
  };

  if (!isIntArray(addrSpaces))
    return emitError() << "addr_space must be integer arrays";
  if (!llvm::all_of<ArrayRef<ArrayAttr>>(
          {accessQuals, types, baseTypes, typeQuals}, isStrArray))
    return emitError()
           << "access_qual, type, base_type, type_qual must be string arrays";
  if (argNames && !isStrArray(argNames)) {
    return emitError() << "name must be a string array";
  }

  if (!llvm::all_of<ArrayRef<ArrayAttr>>(
          {addrSpaces, accessQuals, types, baseTypes, typeQuals, argNames},
          [&](ArrayAttr arr) {
            return !arr || arr.size() == addrSpaces.size();
          })) {
    return emitError() << "all arrays must have the same number of elements";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AddressSpaceAttr definitions
//===----------------------------------------------------------------------===//

std::optional<int32_t>
AddressSpaceAttr::getValueFromLangAS(clang::LangAS langAS) {
  using clang::LangAS;
  switch (langAS) {
  case LangAS::Default:
    // Default address space should be encoded as a null attribute.
    return std::nullopt;
  case LangAS::opencl_global:
    return Kind::offload_global;
  case LangAS::opencl_local:
    return Kind::offload_local;
  case LangAS::opencl_constant:
    return Kind::offload_constant;
  case LangAS::opencl_private:
    return Kind::offload_private;
  case LangAS::opencl_generic:
    return Kind::offload_generic;

  case LangAS::opencl_global_device:
  case LangAS::opencl_global_host:
  case LangAS::cuda_device:
  case LangAS::cuda_constant:
  case LangAS::cuda_shared:
  case LangAS::sycl_global:
  case LangAS::sycl_global_device:
  case LangAS::sycl_global_host:
  case LangAS::sycl_local:
  case LangAS::sycl_private:
  case LangAS::ptr32_sptr:
  case LangAS::ptr32_uptr:
  case LangAS::ptr64:
  case LangAS::hlsl_groupshared:
  case LangAS::wasm_funcref:
    llvm_unreachable("NYI");
  default:
    // Target address space offset arithmetics
    return clang::toTargetAddressSpace(langAS) + kFirstTargetASValue;
  }
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
