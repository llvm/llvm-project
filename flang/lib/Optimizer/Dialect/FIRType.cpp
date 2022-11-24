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
#include "flang/Optimizer/Support/KindMapping.h"
#include "flang/Tools/PointerModels.h"
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
TYPE parseIntSingleton(mlir::AsmParser &parser) {
  int kind = 0;
  if (parser.parseLess() || parser.parseInteger(kind) || parser.parseGreater())
    return {};
  return TYPE::get(parser.getContext(), kind);
}

template <typename TYPE>
TYPE parseKindSingleton(mlir::AsmParser &parser) {
  return parseIntSingleton<TYPE>(parser);
}

template <typename TYPE>
TYPE parseRankSingleton(mlir::AsmParser &parser) {
  return parseIntSingleton<TYPE>(parser);
}

template <typename TYPE>
TYPE parseTypeSingleton(mlir::AsmParser &parser) {
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
  return !(ty.isa<BoxCharType>() || ty.isa<BoxProcType>() ||
           ty.isa<ShapeType>() || ty.isa<ShapeShiftType>() ||
           ty.isa<ShiftType>() || ty.isa<SliceType>() || ty.isa<FieldType>() ||
           ty.isa<LenType>() || ty.isa<ReferenceType>() ||
           ty.isa<TypeDescType>());
}

bool verifySameLists(llvm::ArrayRef<RecordType::TypePair> a1,
                     llvm::ArrayRef<RecordType::TypePair> a2) {
  // FIXME: do we need to allow for any variance here?
  return a1 == a2;
}

RecordType verifyDerived(mlir::AsmParser &parser, RecordType derivedTy,
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
  mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &typeTag, genType);
  if (parseResult.has_value())
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

mlir::Type dyn_cast_ptrEleTy(mlir::Type t) {
  return llvm::TypeSwitch<mlir::Type, mlir::Type>(t)
      .Case<fir::ReferenceType, fir::PointerType, fir::HeapType,
            fir::LLVMPointerType>([](auto p) { return p.getEleTy(); })
      .Default([](mlir::Type) { return mlir::Type{}; });
}

mlir::Type dyn_cast_ptrOrBoxEleTy(mlir::Type t) {
  return llvm::TypeSwitch<mlir::Type, mlir::Type>(t)
      .Case<fir::ReferenceType, fir::PointerType, fir::HeapType,
            fir::LLVMPointerType>([](auto p) { return p.getEleTy(); })
      .Case<fir::BaseBoxType>([](auto p) {
        auto eleTy = p.getEleTy();
        if (auto ty = fir::dyn_cast_ptrEleTy(eleTy))
          return ty;
        return eleTy;
      })
      .Default([](mlir::Type) { return mlir::Type{}; });
}

static bool hasDynamicSize(fir::RecordType recTy) {
  for (auto field : recTy.getTypeList()) {
    if (auto arr = field.second.dyn_cast<fir::SequenceType>()) {
      if (sequenceWithNonConstantShape(arr))
        return true;
    } else if (characterWithDynamicLen(field.second)) {
      return true;
    } else if (auto rec = field.second.dyn_cast<fir::RecordType>()) {
      if (hasDynamicSize(rec))
        return true;
    }
  }
  return false;
}

bool hasDynamicSize(mlir::Type t) {
  if (auto arr = t.dyn_cast<fir::SequenceType>()) {
    if (sequenceWithNonConstantShape(arr))
      return true;
    t = arr.getEleTy();
  }
  if (characterWithDynamicLen(t))
    return true;
  if (auto rec = t.dyn_cast<fir::RecordType>())
    return hasDynamicSize(rec);
  return false;
}

bool isPointerType(mlir::Type ty) {
  if (auto refTy = fir::dyn_cast_ptrEleTy(ty))
    ty = refTy;
  if (auto boxTy = ty.dyn_cast<fir::BaseBoxType>())
    return boxTy.getEleTy().isa<fir::PointerType>();
  return false;
}

bool isAllocatableType(mlir::Type ty) {
  if (auto refTy = fir::dyn_cast_ptrEleTy(ty))
    ty = refTy;
  if (auto boxTy = ty.dyn_cast<fir::BaseBoxType>())
    return boxTy.getEleTy().isa<fir::HeapType>();
  return false;
}

bool isBoxedRecordType(mlir::Type ty) {
  if (auto refTy = fir::dyn_cast_ptrEleTy(ty))
    ty = refTy;
  if (auto boxTy = ty.dyn_cast<fir::BoxType>()) {
    if (boxTy.getEleTy().isa<fir::RecordType>())
      return true;
    mlir::Type innerType = boxTy.unwrapInnerType();
    return innerType && innerType.isa<fir::RecordType>();
  }
  return false;
}

static bool isAssumedType(mlir::Type ty) {
  if (auto boxTy = ty.dyn_cast<fir::BoxType>()) {
    if (boxTy.getEleTy().isa<mlir::NoneType>())
      return true;
    if (auto seqTy = boxTy.getEleTy().dyn_cast<fir::SequenceType>())
      return seqTy.getEleTy().isa<mlir::NoneType>();
  }
  return false;
}

bool isPolymorphicType(mlir::Type ty) {
  if (auto refTy = fir::dyn_cast_ptrEleTy(ty))
    ty = refTy;
  // CLASS(*)
  if (ty.isa<fir::ClassType>())
    return true;
  // assumed type are polymorphic.
  return isAssumedType(ty);
}

bool isUnlimitedPolymorphicType(mlir::Type ty) {
  if (auto refTy = fir::dyn_cast_ptrEleTy(ty))
    ty = refTy;
  // CLASS(*)
  if (auto clTy = ty.dyn_cast<fir::ClassType>()) {
    if (clTy.getEleTy().isa<mlir::NoneType>())
      return true;
    mlir::Type innerType = clTy.unwrapInnerType();
    return innerType && innerType.isa<mlir::NoneType>();
  }
  // TYPE(*)
  return isAssumedType(ty);
}

mlir::Type unwrapInnerType(mlir::Type ty) {
  return llvm::TypeSwitch<mlir::Type, mlir::Type>(ty)
      .Case<fir::PointerType, fir::HeapType, fir::SequenceType>([](auto t) {
        mlir::Type eleTy = t.getEleTy();
        if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>())
          return seqTy.getEleTy();
        return eleTy;
      })
      .Default([](mlir::Type) { return mlir::Type{}; });
}

bool isRecordWithAllocatableMember(mlir::Type ty) {
  if (auto recTy = ty.dyn_cast<fir::RecordType>())
    for (auto [field, memTy] : recTy.getTypeList()) {
      if (fir::isAllocatableType(memTy))
        return true;
      // A record type cannot recursively include itself as a direct member.
      // There must be an intervening `ptr` type, so recursion is safe here.
      if (memTy.isa<fir::RecordType>() && isRecordWithAllocatableMember(memTy))
        return true;
    }
  return false;
}

mlir::Type unwrapAllRefAndSeqType(mlir::Type ty) {
  while (true) {
    mlir::Type nt = unwrapSequenceType(unwrapRefType(ty));
    if (auto vecTy = nt.dyn_cast<fir::VectorType>())
      nt = vecTy.getEleTy();
    if (nt == ty)
      return ty;
    ty = nt;
  }
}

mlir::Type unwrapSeqOrBoxedSeqType(mlir::Type ty) {
  if (auto seqTy = ty.dyn_cast<fir::SequenceType>())
    return seqTy.getEleTy();
  if (auto boxTy = ty.dyn_cast<fir::BoxType>()) {
    auto eleTy = unwrapRefType(boxTy.getEleTy());
    if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>())
      return seqTy.getEleTy();
  }
  return ty;
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
mlir::Type BoxProcType::parse(mlir::AsmParser &parser) {
  mlir::Type ty;
  if (parser.parseLess() || parser.parseType(ty) || parser.parseGreater())
    return {};
  return get(parser.getContext(), ty);
}

void fir::BoxProcType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getEleTy() << '>';
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

static bool cannotBePointerOrHeapElementType(mlir::Type eleTy) {
  return eleTy.isa<BoxType, BoxCharType, BoxProcType, ShapeType, ShapeShiftType,
                   SliceType, FieldType, LenType, HeapType, PointerType,
                   ReferenceType, TypeDescType>();
}

//===----------------------------------------------------------------------===//
// BoxType
//===----------------------------------------------------------------------===//

mlir::LogicalResult
fir::BoxType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                     mlir::Type eleTy) {
  // TODO
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BoxCharType
//===----------------------------------------------------------------------===//

mlir::Type fir::BoxCharType::parse(mlir::AsmParser &parser) {
  return parseKindSingleton<fir::BoxCharType>(parser);
}

void fir::BoxCharType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getKind() << ">";
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
mlir::Type fir::CharacterType::parse(mlir::AsmParser &parser) {
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
  return get(parser.getContext(), kind, len);
}

void fir::CharacterType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getFKind();
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
// ClassType
//===----------------------------------------------------------------------===//

mlir::LogicalResult
fir::ClassType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                       mlir::Type eleTy) {
  if (eleTy.isa<fir::RecordType, fir::SequenceType, fir::HeapType,
                fir::PointerType, mlir::NoneType, mlir::IntegerType,
                mlir::FloatType>())
    return mlir::success();
  return emitError() << "invalid element type\n";
}

//===----------------------------------------------------------------------===//
// ComplexType
//===----------------------------------------------------------------------===//

mlir::Type fir::ComplexType::parse(mlir::AsmParser &parser) {
  return parseKindSingleton<fir::ComplexType>(parser);
}

void fir::ComplexType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getFKind() << '>';
}

mlir::Type fir::ComplexType::getElementType() const {
  return fir::RealType::get(getContext(), getFKind());
}

// Return the MLIR float type of the complex element type.
mlir::Type fir::ComplexType::getEleType(const fir::KindMapping &kindMap) const {
  auto fkind = getFKind();
  auto realTypeID = kindMap.getRealTypeID(fkind);
  return fir::fromRealTypeID(getContext(), realTypeID, fkind);
}

//===----------------------------------------------------------------------===//
// HeapType
//===----------------------------------------------------------------------===//

// `heap` `<` type `>`
mlir::Type fir::HeapType::parse(mlir::AsmParser &parser) {
  return parseTypeSingleton<HeapType>(parser);
}

void fir::HeapType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getEleTy() << '>';
}

mlir::LogicalResult
fir::HeapType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                      mlir::Type eleTy) {
  if (cannotBePointerOrHeapElementType(eleTy))
    return emitError() << "cannot build a heap pointer to type: " << eleTy
                       << '\n';
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// IntegerType
//===----------------------------------------------------------------------===//

// `int` `<` kind `>`
mlir::Type fir::IntegerType::parse(mlir::AsmParser &parser) {
  return parseKindSingleton<fir::IntegerType>(parser);
}

void fir::IntegerType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getFKind() << '>';
}

//===----------------------------------------------------------------------===//
// LogicalType
//===----------------------------------------------------------------------===//

// `logical` `<` kind `>`
mlir::Type fir::LogicalType::parse(mlir::AsmParser &parser) {
  return parseKindSingleton<fir::LogicalType>(parser);
}

void fir::LogicalType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getFKind() << '>';
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

// `ptr` `<` type `>`
mlir::Type fir::PointerType::parse(mlir::AsmParser &parser) {
  return parseTypeSingleton<fir::PointerType>(parser);
}

void fir::PointerType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getEleTy() << '>';
}

mlir::LogicalResult fir::PointerType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type eleTy) {
  if (cannotBePointerOrHeapElementType(eleTy))
    return emitError() << "cannot build a pointer to type: " << eleTy << '\n';
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RealType
//===----------------------------------------------------------------------===//

// `real` `<` kind `>`
mlir::Type fir::RealType::parse(mlir::AsmParser &parser) {
  return parseKindSingleton<fir::RealType>(parser);
}

void fir::RealType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getFKind() << '>';
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
mlir::Type fir::RecordType::parse(mlir::AsmParser &parser) {
  llvm::StringRef name;
  if (parser.parseLess() || parser.parseKeyword(&name))
    return {};
  RecordType result = RecordType::get(parser.getContext(), name);

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

void fir::RecordType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getName();
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
mlir::Type fir::ReferenceType::parse(mlir::AsmParser &parser) {
  return parseTypeSingleton<fir::ReferenceType>(parser);
}

void fir::ReferenceType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getEleTy() << '>';
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
mlir::Type fir::SequenceType::parse(mlir::AsmParser &parser) {
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
  if (parser.parseType(eleTy))
    return {};
  mlir::AffineMapAttr map;
  if (!parser.parseOptionalComma()) {
    if (parser.parseAttribute(map)) {
      parser.emitError(parser.getNameLoc(), "expecting affine map");
      return {};
    }
  }
  if (parser.parseGreater())
    return {};
  return SequenceType::get(parser.getContext(), shape, eleTy, map);
}

void fir::SequenceType::print(mlir::AsmPrinter &printer) const {
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
  if (hasDynamicSize(getEleTy()))
    return 0;
  auto shape = getShape();
  unsigned count = 0;
  for (auto d : shape) {
    if (d == getUnknownExtent())
      break;
    ++count;
  }
  return count;
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

mlir::Type fir::ShapeType::parse(mlir::AsmParser &parser) {
  return parseRankSingleton<fir::ShapeType>(parser);
}

void fir::ShapeType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getImpl()->rank << ">";
}

//===----------------------------------------------------------------------===//
// ShapeShiftType
//===----------------------------------------------------------------------===//

mlir::Type fir::ShapeShiftType::parse(mlir::AsmParser &parser) {
  return parseRankSingleton<fir::ShapeShiftType>(parser);
}

void fir::ShapeShiftType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getRank() << ">";
}

//===----------------------------------------------------------------------===//
// ShiftType
//===----------------------------------------------------------------------===//

mlir::Type fir::ShiftType::parse(mlir::AsmParser &parser) {
  return parseRankSingleton<fir::ShiftType>(parser);
}

void fir::ShiftType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getRank() << ">";
}

//===----------------------------------------------------------------------===//
// SliceType
//===----------------------------------------------------------------------===//

// `slice` `<` rank `>`
mlir::Type fir::SliceType::parse(mlir::AsmParser &parser) {
  return parseRankSingleton<fir::SliceType>(parser);
}

void fir::SliceType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getRank() << '>';
}

//===----------------------------------------------------------------------===//
// TypeDescType
//===----------------------------------------------------------------------===//

// `tdesc` `<` type `>`
mlir::Type fir::TypeDescType::parse(mlir::AsmParser &parser) {
  return parseTypeSingleton<fir::TypeDescType>(parser);
}

void fir::TypeDescType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getOfTy() << '>';
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
mlir::Type fir::VectorType::parse(mlir::AsmParser &parser) {
  int64_t len = 0;
  mlir::Type eleTy;
  if (parser.parseLess() || parser.parseInteger(len) || parser.parseColon() ||
      parser.parseType(eleTy) || parser.parseGreater())
    return {};
  return fir::VectorType::get(len, eleTy);
}

void fir::VectorType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getLen() << ':' << getEleTy() << '>';
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

bool fir::isCharacterProcedureTuple(mlir::Type ty, bool acceptRawFunc) {
  mlir::TupleType tuple = ty.dyn_cast<mlir::TupleType>();
  return tuple && tuple.size() == 2 &&
         (tuple.getType(0).isa<fir::BoxProcType>() ||
          (acceptRawFunc && tuple.getType(0).isa<mlir::FunctionType>())) &&
         fir::isa_integer(tuple.getType(1));
}

bool fir::hasAbstractResult(mlir::FunctionType ty) {
  if (ty.getNumResults() == 0)
    return false;
  auto resultType = ty.getResult(0);
  return resultType.isa<fir::SequenceType, fir::BoxType, fir::RecordType>();
}

/// Convert llvm::Type::TypeID to mlir::Type. \p kind is provided for error
/// messages only.
mlir::Type fir::fromRealTypeID(mlir::MLIRContext *context,
                               llvm::Type::TypeID typeID, fir::KindTy kind) {
  switch (typeID) {
  case llvm::Type::TypeID::HalfTyID:
    return mlir::FloatType::getF16(context);
  case llvm::Type::TypeID::BFloatTyID:
    return mlir::FloatType::getBF16(context);
  case llvm::Type::TypeID::FloatTyID:
    return mlir::FloatType::getF32(context);
  case llvm::Type::TypeID::DoubleTyID:
    return mlir::FloatType::getF64(context);
  case llvm::Type::TypeID::X86_FP80TyID:
    return mlir::FloatType::getF80(context);
  case llvm::Type::TypeID::FP128TyID:
    return mlir::FloatType::getF128(context);
  default:
    mlir::emitError(mlir::UnknownLoc::get(context))
        << "unsupported type: !fir.real<" << kind << ">";
    return {};
  }
}

//===----------------------------------------------------------------------===//
// BaseBoxType
//===----------------------------------------------------------------------===//

mlir::Type BaseBoxType::getEleTy() const {
  return llvm::TypeSwitch<fir::BaseBoxType, mlir::Type>(*this)
      .Case<fir::BoxType, fir::ClassType>(
          [](auto type) { return type.getEleTy(); });
}

mlir::Type BaseBoxType::unwrapInnerType() const {
  return fir::unwrapInnerType(getEleTy());
}

//===----------------------------------------------------------------------===//
// FIROpsDialect
//===----------------------------------------------------------------------===//

void FIROpsDialect::registerTypes() {
  addTypes<BoxType, BoxCharType, BoxProcType, CharacterType, ClassType,
           fir::ComplexType, FieldType, HeapType, fir::IntegerType, LenType,
           LogicalType, LLVMPointerType, PointerType, RealType, RecordType,
           ReferenceType, SequenceType, ShapeType, ShapeShiftType, ShiftType,
           SliceType, TypeDescType, fir::VectorType>();
  fir::ReferenceType::attachInterface<PointerLikeModel<fir::ReferenceType>>(
      *getContext());

  fir::PointerType::attachInterface<PointerLikeModel<fir::PointerType>>(
      *getContext());

  fir::HeapType::attachInterface<AlternativePointerLikeModel<fir::HeapType>>(
      *getContext());

  fir::LLVMPointerType::attachInterface<
      AlternativePointerLikeModel<fir::LLVMPointerType>>(*getContext());
}
