//===-- FIRType.cpp -------------------------------------------------------===//
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

#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_TYPEDEF_CLASSES
#include "flang/Optimizer/Dialect/FIROpsTypes.cpp.inc"

using namespace fir;

namespace {

template <typename TYPE>
TYPE parseIntSingleton(mlir::DialectAsmParser &parser) {
  int kind = 0;
  if (parser.parseLess() || parser.parseInteger(kind) || parser.parseGreater())
    return {};
  return TYPE::get(parser.getBuilder().getContext(), kind);
}

template <typename TYPE>
TYPE parseKindSingleton(mlir::DialectAsmParser &parser) {
  return parseIntSingleton<TYPE>(parser);
}

template <typename TYPE>
TYPE parseRankSingleton(mlir::DialectAsmParser &parser) {
  return parseIntSingleton<TYPE>(parser);
}

template <typename TYPE>
TYPE parseTypeSingleton(mlir::DialectAsmParser &parser) {
  mlir::Type ty;
  if (parser.parseLess() || parser.parseType(ty) || parser.parseGreater())
    return {};
  return TYPE::get(ty);
}

/// Is `ty` a standard or FIR integer type?
static bool isaIntegerType(mlir::Type ty) {
  // TODO: why aren't we using isa_integer? investigatation required.
  return ty.isa<mlir::IntegerType>() || ty.isa<fir::IntegerType>();
}

bool verifyRecordMemberType(mlir::Type ty) {
  return !(ty.isa<BoxType>() || ty.isa<BoxCharType>() ||
           ty.isa<BoxProcType>() || ty.isa<ShapeType>() ||
           ty.isa<ShapeShiftType>() || ty.isa<ShiftType>() ||
           ty.isa<SliceType>() || ty.isa<FieldType>() || ty.isa<LenType>() ||
           ty.isa<ReferenceType>() || ty.isa<TypeDescType>());
}

bool verifySameLists(llvm::ArrayRef<RecordType::TypePair> a1,
                     llvm::ArrayRef<RecordType::TypePair> a2) {
  // FIXME: do we need to allow for any variance here?
  return a1 == a2;
}

RecordType verifyDerived(mlir::DialectAsmParser &parser, RecordType derivedTy,
                         llvm::ArrayRef<RecordType::TypePair> lenPList,
                         llvm::ArrayRef<RecordType::TypePair> typeList) {
  auto loc = parser.getNameLoc();
  if (!verifySameLists(derivedTy.getLenParamList(), lenPList) ||
      !verifySameLists(derivedTy.getTypeList(), typeList)) {
    parser.emitError(loc, "cannot redefine record type members");
    return {};
  }
  for (auto &p : lenPList)
    if (!isaIntegerType(p.second)) {
      parser.emitError(loc, "LEN parameter must be integral type");
      return {};
    }
  for (auto &p : typeList)
    if (!verifyRecordMemberType(p.second)) {
      parser.emitError(loc, "field parameter has invalid type");
      return {};
    }
  llvm::StringSet<> uniq;
  for (auto &p : lenPList)
    if (!uniq.insert(p.first).second) {
      parser.emitError(loc, "LEN parameter cannot have duplicate name");
      return {};
    }
  for (auto &p : typeList)
    if (!uniq.insert(p.first).second) {
      parser.emitError(loc, "field cannot have duplicate name");
      return {};
    }
  return derivedTy;
}

} // namespace

// Implementation of the thin interface from dialect to type parser

mlir::Type fir::parseFirType(FIROpsDialect *dialect,
                             mlir::DialectAsmParser &parser) {
  mlir::StringRef typeTag;
  if (parser.parseKeyword(&typeTag))
    return {};
  mlir::Type genType;
  auto parseResult = generatedTypeParser(parser.getBuilder().getContext(),
                                         parser, typeTag, genType);
  if (parseResult.hasValue())
    return genType;
  parser.emitError(parser.getNameLoc(), "unknown fir type: ") << typeTag;
  return {};
}

namespace fir {
namespace detail {

// Type storage classes

/// Derived type storage
struct RecordTypeStorage : public mlir::TypeStorage {
  using KeyTy = llvm::StringRef;

  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.str());
  }

  bool operator==(const KeyTy &key) const { return key == getName(); }

  static RecordTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    auto *storage = allocator.allocate<RecordTypeStorage>();
    return new (storage) RecordTypeStorage{key};
  }

  llvm::StringRef getName() const { return name; }

  void setLenParamList(llvm::ArrayRef<RecordType::TypePair> list) {
    lens = list;
  }
  llvm::ArrayRef<RecordType::TypePair> getLenParamList() const { return lens; }

  void setTypeList(llvm::ArrayRef<RecordType::TypePair> list) { types = list; }
  llvm::ArrayRef<RecordType::TypePair> getTypeList() const { return types; }

  void finalize(llvm::ArrayRef<RecordType::TypePair> lenParamList,
                llvm::ArrayRef<RecordType::TypePair> typeList) {
    if (finalized)
      return;
    finalized = true;
    setLenParamList(lenParamList);
    setTypeList(typeList);
  }

protected:
  std::string name;
  bool finalized;
  std::vector<RecordType::TypePair> lens;
  std::vector<RecordType::TypePair> types;

private:
  RecordTypeStorage() = delete;
  explicit RecordTypeStorage(llvm::StringRef name)
      : name{name}, finalized{false} {}
};

} // namespace detail

template <typename A, typename B>
bool inbounds(A v, B lb, B ub) {
  return v >= lb && v < ub;
}

bool isa_fir_type(mlir::Type t) {
  return llvm::isa<FIROpsDialect>(t.getDialect());
}

bool isa_std_type(mlir::Type t) {
  return llvm::isa<mlir::BuiltinDialect>(t.getDialect());
}

bool isa_fir_or_std_type(mlir::Type t) {
  if (auto funcType = t.dyn_cast<mlir::FunctionType>())
    return llvm::all_of(funcType.getInputs(), isa_fir_or_std_type) &&
           llvm::all_of(funcType.getResults(), isa_fir_or_std_type);
  return isa_fir_type(t) || isa_std_type(t);
}

bool isa_ref_type(mlir::Type t) {
  return t.isa<ReferenceType>() || t.isa<PointerType>() || t.isa<HeapType>();
}

bool isa_box_type(mlir::Type t) {
  return t.isa<BoxType>() || t.isa<BoxCharType>() || t.isa<BoxProcType>();
}

bool isa_passbyref_type(mlir::Type t) {
  return t.isa<ReferenceType>() || isa_box_type(t) ||
         t.isa<mlir::FunctionType>();
}

bool isa_aggregate(mlir::Type t) {
  return t.isa<SequenceType>() || t.isa<RecordType>() ||
         t.isa<mlir::TupleType>();
}

mlir::Type dyn_cast_ptrEleTy(mlir::Type t) {
  return llvm::TypeSwitch<mlir::Type, mlir::Type>(t)
      .Case<fir::ReferenceType, fir::PointerType, fir::HeapType>(
          [](auto p) { return p.getEleTy(); })
      .Default([](mlir::Type) { return mlir::Type{}; });
}

mlir::Type dyn_cast_ptrOrBoxEleTy(mlir::Type t) {
  return llvm::TypeSwitch<mlir::Type, mlir::Type>(t)
      .Case<fir::ReferenceType, fir::PointerType, fir::HeapType>(
          [](auto p) { return p.getEleTy(); })
      .Case<fir::BoxType>([](auto p) {
        auto eleTy = p.getEleTy();
        if (auto ty = fir::dyn_cast_ptrEleTy(eleTy))
          return ty;
        return eleTy;
      })
      .Default([](mlir::Type) { return mlir::Type{}; });
}

} // namespace fir

namespace {

static llvm::SmallPtrSet<detail::RecordTypeStorage const *, 4>
    recordTypeVisited;

} // namespace

void fir::verifyIntegralType(mlir::Type type) {
  if (isaIntegerType(type) || type.isa<mlir::IndexType>())
    return;
  llvm::report_fatal_error("expected integral type");
}

void fir::printFirType(FIROpsDialect *, mlir::Type ty,
                       mlir::DialectAsmPrinter &p) {
  if (mlir::failed(generatedTypePrinter(ty, p)))
    llvm::report_fatal_error("unknown type to print");
}

bool fir::isa_unknown_size_box(mlir::Type t) {
  if (auto boxTy = t.dyn_cast<fir::BoxType>()) {
    auto eleTy = boxTy.getEleTy();
    if (auto actualEleTy = fir::dyn_cast_ptrEleTy(eleTy))
      eleTy = actualEleTy;
    if (eleTy.isa<mlir::NoneType>())
      return true;
    if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>())
      if (seqTy.hasUnknownShape())
        return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// BoxProcType
//===----------------------------------------------------------------------===//

// `boxproc` `<` return-type `>`
mlir::Type BoxProcType::parse(mlir::MLIRContext *context,
                              mlir::DialectAsmParser &parser) {
  mlir::Type ty;
  if (parser.parseLess() || parser.parseType(ty) || parser.parseGreater())
    return {};
  return get(context, ty);
}

void fir::BoxProcType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getEleTy() << '>';
}

mlir::LogicalResult
BoxProcType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                    mlir::Type eleTy) {
  if (eleTy.isa<mlir::FunctionType>())
    return mlir::success();
  if (auto refTy = eleTy.dyn_cast<ReferenceType>())
    if (refTy.isa<mlir::FunctionType>())
      return mlir::success();
  return emitError() << "invalid type for boxproc" << eleTy << '\n';
}

static bool canBePointerOrHeapElementType(mlir::Type eleTy) {
  return eleTy.isa<BoxType, BoxCharType, BoxProcType, ShapeType, ShapeShiftType,
                   SliceType, FieldType, LenType, HeapType, PointerType,
                   ReferenceType, TypeDescType>();
}

//===----------------------------------------------------------------------===//
// BoxType
//===----------------------------------------------------------------------===//

// `box` `<` type (',' affine-map)? `>`
mlir::Type fir::BoxType::parse(mlir::MLIRContext *context,
                               mlir::DialectAsmParser &parser) {
  mlir::Type ofTy;
  if (parser.parseLess() || parser.parseType(ofTy))
    return {};

  mlir::AffineMapAttr map;
  if (!parser.parseOptionalComma()) {
    if (parser.parseAttribute(map)) {
      parser.emitError(parser.getCurrentLocation(), "expected affine map");
      return {};
    }
  }
  if (parser.parseGreater())
    return {};
  return get(ofTy, map);
}

void fir::BoxType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getEleTy();
  if (auto map = getLayoutMap()) {
    printer << ", " << map;
  }
  printer << '>';
}

mlir::LogicalResult
fir::BoxType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                     mlir::Type eleTy, mlir::AffineMapAttr map) {
  // TODO
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BoxCharType
//===----------------------------------------------------------------------===//

mlir::Type fir::BoxCharType::parse(mlir::MLIRContext *context,
                                   mlir::DialectAsmParser &parser) {
  return parseKindSingleton<fir::BoxCharType>(parser);
}

void fir::BoxCharType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getKind() << ">";
}

CharacterType
fir::BoxCharType::getElementType(mlir::MLIRContext *context) const {
  return CharacterType::getUnknownLen(context, getKind());
}

CharacterType fir::BoxCharType::getEleTy() const {
  return getElementType(getContext());
}

//===----------------------------------------------------------------------===//
// CharacterType
//===----------------------------------------------------------------------===//

// `char` `<` kind [`,` `len`] `>`
mlir::Type fir::CharacterType::parse(mlir::MLIRContext *context,
                                     mlir::DialectAsmParser &parser) {
  int kind = 0;
  if (parser.parseLess() || parser.parseInteger(kind))
    return {};
  CharacterType::LenType len = 1;
  if (mlir::succeeded(parser.parseOptionalComma())) {
    if (mlir::succeeded(parser.parseOptionalQuestion())) {
      len = fir::CharacterType::unknownLen();
    } else if (!mlir::succeeded(parser.parseInteger(len))) {
      return {};
    }
  }
  if (parser.parseGreater())
    return {};
  return get(context, kind, len);
}

void fir::CharacterType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getFKind();
  auto len = getLen();
  if (len != fir::CharacterType::singleton()) {
    printer << ',';
    if (len == fir::CharacterType::unknownLen())
      printer << '?';
    else
      printer << len;
  }
  printer << '>';
}

//===----------------------------------------------------------------------===//
// ComplexType
//===----------------------------------------------------------------------===//

mlir::Type fir::ComplexType::parse(mlir::MLIRContext *context,
                                   mlir::DialectAsmParser &parser) {
  return parseKindSingleton<fir::ComplexType>(parser);
}

void fir::ComplexType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getFKind() << '>';
}

mlir::Type fir::ComplexType::getElementType() const {
  return fir::RealType::get(getContext(), getFKind());
}

//===----------------------------------------------------------------------===//
// HeapType
//===----------------------------------------------------------------------===//

// `heap` `<` type `>`
mlir::Type fir::HeapType::parse(mlir::MLIRContext *context,
                                mlir::DialectAsmParser &parser) {
  return parseTypeSingleton<HeapType>(parser);
}

void fir::HeapType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getEleTy() << '>';
}

mlir::LogicalResult
fir::HeapType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                      mlir::Type eleTy) {
  if (canBePointerOrHeapElementType(eleTy))
    return emitError() << "cannot build a heap pointer to type: " << eleTy
                       << '\n';
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// IntegerType
//===----------------------------------------------------------------------===//

// `int` `<` kind `>`
mlir::Type fir::IntegerType::parse(mlir::MLIRContext *context,
                                   mlir::DialectAsmParser &parser) {
  return parseKindSingleton<fir::IntegerType>(parser);
}

void fir::IntegerType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getFKind() << '>';
}

//===----------------------------------------------------------------------===//
// LogicalType
//===----------------------------------------------------------------------===//

// `logical` `<` kind `>`
mlir::Type fir::LogicalType::parse(mlir::MLIRContext *context,
                                   mlir::DialectAsmParser &parser) {
  return parseKindSingleton<fir::LogicalType>(parser);
}

void fir::LogicalType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getFKind() << '>';
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

// `ptr` `<` type `>`
mlir::Type fir::PointerType::parse(mlir::MLIRContext *context,
                                   mlir::DialectAsmParser &parser) {
  return parseTypeSingleton<fir::PointerType>(parser);
}

void fir::PointerType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getEleTy() << '>';
}

mlir::LogicalResult fir::PointerType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type eleTy) {
  if (canBePointerOrHeapElementType(eleTy))
    return emitError() << "cannot build a pointer to type: " << eleTy << '\n';
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RealType
//===----------------------------------------------------------------------===//

// `real` `<` kind `>`
mlir::Type fir::RealType::parse(mlir::MLIRContext *context,
                                mlir::DialectAsmParser &parser) {
  return parseKindSingleton<fir::RealType>(parser);
}

void fir::RealType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getFKind() << '>';
}

mlir::LogicalResult
fir::RealType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                      KindTy fKind) {
  // TODO
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RecordType
//===----------------------------------------------------------------------===//

// Fortran derived type
// `type` `<` name
//           (`(` id `:` type (`,` id `:` type)* `)`)?
//           (`{` id `:` type (`,` id `:` type)* `}`)? '>'
mlir::Type fir::RecordType::parse(mlir::MLIRContext *context,
                                  mlir::DialectAsmParser &parser) {
  llvm::StringRef name;
  if (parser.parseLess() || parser.parseKeyword(&name))
    return {};
  RecordType result = RecordType::get(parser.getBuilder().getContext(), name);

  RecordType::TypeList lenParamList;
  if (!parser.parseOptionalLParen()) {
    while (true) {
      llvm::StringRef lenparam;
      mlir::Type intTy;
      if (parser.parseKeyword(&lenparam) || parser.parseColon() ||
          parser.parseType(intTy)) {
        parser.emitError(parser.getNameLoc(), "expected LEN parameter list");
        return {};
      }
      lenParamList.emplace_back(lenparam, intTy);
      if (parser.parseOptionalComma())
        break;
    }
    if (parser.parseRParen())
      return {};
  }

  RecordType::TypeList typeList;
  if (!parser.parseOptionalLBrace()) {
    while (true) {
      llvm::StringRef field;
      mlir::Type fldTy;
      if (parser.parseKeyword(&field) || parser.parseColon() ||
          parser.parseType(fldTy)) {
        parser.emitError(parser.getNameLoc(), "expected field type list");
        return {};
      }
      typeList.emplace_back(field, fldTy);
      if (parser.parseOptionalComma())
        break;
    }
    if (parser.parseRBrace())
      return {};
  }

  if (parser.parseGreater())
    return {};

  if (lenParamList.empty() && typeList.empty())
    return result;

  result.finalize(lenParamList, typeList);
  return verifyDerived(parser, result, lenParamList, typeList);
}

void fir::RecordType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getName();
  if (!recordTypeVisited.count(uniqueKey())) {
    recordTypeVisited.insert(uniqueKey());
    if (getLenParamList().size()) {
      char ch = '(';
      for (auto p : getLenParamList()) {
        printer << ch << p.first << ':';
        p.second.print(printer.getStream());
        ch = ',';
      }
      printer << ')';
    }
    if (getTypeList().size()) {
      char ch = '{';
      for (auto p : getTypeList()) {
        printer << ch << p.first << ':';
        p.second.print(printer.getStream());
        ch = ',';
      }
      printer << '}';
    }
    recordTypeVisited.erase(uniqueKey());
  }
  printer << '>';
}

void fir::RecordType::finalize(llvm::ArrayRef<TypePair> lenPList,
                               llvm::ArrayRef<TypePair> typeList) {
  getImpl()->finalize(lenPList, typeList);
}

llvm::StringRef fir::RecordType::getName() const {
  return getImpl()->getName();
}

RecordType::TypeList fir::RecordType::getTypeList() const {
  return getImpl()->getTypeList();
}

RecordType::TypeList fir::RecordType::getLenParamList() const {
  return getImpl()->getLenParamList();
}

detail::RecordTypeStorage const *fir::RecordType::uniqueKey() const {
  return getImpl();
}

mlir::LogicalResult fir::RecordType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef name) {
  if (name.size() == 0)
    return emitError() << "record types must have a name";
  return mlir::success();
}

mlir::Type fir::RecordType::getType(llvm::StringRef ident) {
  for (auto f : getTypeList())
    if (ident == f.first)
      return f.second;
  return {};
}

unsigned fir::RecordType::getFieldIndex(llvm::StringRef ident) {
  for (auto f : llvm::enumerate(getTypeList()))
    if (ident == f.value().first)
      return f.index();
  return std::numeric_limits<unsigned>::max();
}

//===----------------------------------------------------------------------===//
// ReferenceType
//===----------------------------------------------------------------------===//

// `ref` `<` type `>`
mlir::Type fir::ReferenceType::parse(mlir::MLIRContext *context,
                                     mlir::DialectAsmParser &parser) {
  return parseTypeSingleton<fir::ReferenceType>(parser);
}

void fir::ReferenceType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getEleTy() << '>';
}

mlir::LogicalResult fir::ReferenceType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type eleTy) {
  if (eleTy.isa<ShapeType, ShapeShiftType, SliceType, FieldType, LenType,
                ReferenceType, TypeDescType>())
    return emitError() << "cannot build a reference to type: " << eleTy << '\n';
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SequenceType
//===----------------------------------------------------------------------===//

// `array` `<` `*` | bounds (`x` bounds)* `:` type (',' affine-map)? `>`
// bounds ::= `?` | int-lit
mlir::Type fir::SequenceType::parse(mlir::MLIRContext *context,
                                    mlir::DialectAsmParser &parser) {
  if (parser.parseLess())
    return {};
  SequenceType::Shape shape;
  if (parser.parseOptionalStar()) {
    if (parser.parseDimensionList(shape, /*allowDynamic=*/true))
      return {};
  } else if (parser.parseColon()) {
    return {};
  }
  mlir::Type eleTy;
  if (parser.parseType(eleTy) || parser.parseGreater())
    return {};
  mlir::AffineMapAttr map;
  if (!parser.parseOptionalComma())
    if (parser.parseAttribute(map)) {
      parser.emitError(parser.getNameLoc(), "expecting affine map");
      return {};
    }
  return SequenceType::get(context, shape, eleTy, map);
}

void fir::SequenceType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic();
  auto shape = getShape();
  if (shape.size()) {
    printer << '<';
    for (const auto &b : shape) {
      if (b >= 0)
        printer << b << 'x';
      else
        printer << "?x";
    }
  } else {
    printer << "<*:";
  }
  printer << getEleTy();
  if (auto map = getLayoutMap()) {
    printer << ", ";
    map.print(printer.getStream());
  }
  printer << '>';
}

unsigned fir::SequenceType::getConstantRows() const {
  auto shape = getShape();
  unsigned count = 0;
  for (auto d : shape) {
    if (d < 0)
      break;
    ++count;
  }
  return count;
}

// This test helps us determine if we can degenerate an array to a
// pointer to some interior section (possibly a single element) of the
// sequence. This is used to determine if we can lower to the LLVM IR.
bool fir::SequenceType::hasConstantInterior() const {
  if (hasUnknownShape())
    return true;
  auto rows = getConstantRows();
  auto dim = getDimension();
  if (rows == dim)
    return true;
  auto shape = getShape();
  for (unsigned i = rows, size = dim; i < size; ++i)
    if (shape[i] != getUnknownExtent())
      return false;
  return true;
}

mlir::LogicalResult fir::SequenceType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, mlir::Type eleTy,
    mlir::AffineMapAttr layoutMap) {
  // DIMENSION attribute can only be applied to an intrinsic or record type
  if (eleTy.isa<BoxType, BoxCharType, BoxProcType, ShapeType, ShapeShiftType,
                ShiftType, SliceType, FieldType, LenType, HeapType, PointerType,
                ReferenceType, TypeDescType, fir::VectorType, SequenceType>())
    return emitError() << "cannot build an array of this element type: "
                       << eleTy << '\n';
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ShapeType
//===----------------------------------------------------------------------===//

mlir::Type fir::ShapeType::parse(mlir::MLIRContext *context,
                                 mlir::DialectAsmParser &parser) {
  return parseRankSingleton<fir::ShapeType>(parser);
}

void fir::ShapeType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getImpl()->rank << ">";
}

//===----------------------------------------------------------------------===//
// ShapeShiftType
//===----------------------------------------------------------------------===//

mlir::Type fir::ShapeShiftType::parse(mlir::MLIRContext *context,
                                      mlir::DialectAsmParser &parser) {
  return parseRankSingleton<fir::ShapeShiftType>(parser);
}

void fir::ShapeShiftType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getRank() << ">";
}

//===----------------------------------------------------------------------===//
// ShiftType
//===----------------------------------------------------------------------===//

mlir::Type fir::ShiftType::parse(mlir::MLIRContext *context,
                                 mlir::DialectAsmParser &parser) {
  return parseRankSingleton<fir::ShiftType>(parser);
}

void fir::ShiftType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getRank() << ">";
}

//===----------------------------------------------------------------------===//
// SliceType
//===----------------------------------------------------------------------===//

// `slice` `<` rank `>`
mlir::Type fir::SliceType::parse(mlir::MLIRContext *context,
                                 mlir::DialectAsmParser &parser) {
  return parseRankSingleton<fir::SliceType>(parser);
}

void fir::SliceType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getRank() << '>';
}

//===----------------------------------------------------------------------===//
// TypeDescType
//===----------------------------------------------------------------------===//

// `tdesc` `<` type `>`
mlir::Type fir::TypeDescType::parse(mlir::MLIRContext *context,
                                    mlir::DialectAsmParser &parser) {
  return parseTypeSingleton<fir::TypeDescType>(parser);
}

void fir::TypeDescType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getOfTy() << '>';
}

mlir::LogicalResult fir::TypeDescType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type eleTy) {
  if (eleTy.isa<BoxType, BoxCharType, BoxProcType, ShapeType, ShapeShiftType,
                ShiftType, SliceType, FieldType, LenType, ReferenceType,
                TypeDescType>())
    return emitError() << "cannot build a type descriptor of type: " << eleTy
                       << '\n';
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// VectorType
//===----------------------------------------------------------------------===//

// `vector` `<` len `:` type `>`
mlir::Type fir::VectorType::parse(mlir::MLIRContext *context,
                                  mlir::DialectAsmParser &parser) {
  int64_t len = 0;
  mlir::Type eleTy;
  if (parser.parseLess() || parser.parseInteger(len) || parser.parseColon() ||
      parser.parseType(eleTy) || parser.parseGreater())
    return {};
  return fir::VectorType::get(len, eleTy);
}

void fir::VectorType::print(mlir::DialectAsmPrinter &printer) const {
  printer << getMnemonic() << "<" << getLen() << ':' << getEleTy() << '>';
}

mlir::LogicalResult fir::VectorType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, uint64_t len,
    mlir::Type eleTy) {
  if (!(fir::isa_real(eleTy) || fir::isa_integer(eleTy)))
    return emitError() << "cannot build a vector of type " << eleTy << '\n';
  return mlir::success();
}

bool fir::VectorType::isValidElementType(mlir::Type t) {
  return isa_real(t) || isa_integer(t);
}

//===----------------------------------------------------------------------===//
// FIROpsDialect
//===----------------------------------------------------------------------===//

void FIROpsDialect::registerTypes() {
  addTypes<BoxType, BoxCharType, BoxProcType, CharacterType, fir::ComplexType,
           FieldType, HeapType, fir::IntegerType, LenType, LogicalType,
           PointerType, RealType, RecordType, ReferenceType, SequenceType,
           ShapeType, ShapeShiftType, ShiftType, SliceType, TypeDescType,
           fir::VectorType>();
}
