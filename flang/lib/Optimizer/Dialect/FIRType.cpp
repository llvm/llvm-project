//===-- FIRType.cpp -------------------------------------------------------===//
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

#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Common/ISO_Fortran_binding_wrapper.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Tools/PointerModels.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinDialect.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/DialectImplementation.h"
#include "aiir/Support/LLVM.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_TYPEDEF_CLASSES
#include "flang/Optimizer/Dialect/FIROpsTypes.cpp.inc"

using namespace fir;

namespace {

template <typename TYPE>
TYPE parseIntSingleton(aiir::AsmParser &parser) {
  int kind = 0;
  if (parser.parseLess() || parser.parseInteger(kind) || parser.parseGreater())
    return {};
  return TYPE::get(parser.getContext(), kind);
}

template <typename TYPE>
TYPE parseKindSingleton(aiir::AsmParser &parser) {
  return parseIntSingleton<TYPE>(parser);
}

template <typename TYPE>
TYPE parseRankSingleton(aiir::AsmParser &parser) {
  return parseIntSingleton<TYPE>(parser);
}

template <typename TYPE>
TYPE parseTypeSingleton(aiir::AsmParser &parser) {
  aiir::Type ty;
  if (parser.parseLess() || parser.parseType(ty) || parser.parseGreater())
    return {};
  return TYPE::get(ty);
}

/// Is `ty` a standard or FIR integer type?
static bool isaIntegerType(aiir::Type ty) {
  // TODO: why aren't we using isa_integer? investigatation required.
  return aiir::isa<aiir::IntegerType, fir::IntegerType>(ty);
}

bool verifyRecordMemberType(aiir::Type ty) {
  return !aiir::isa<BoxCharType, ShapeType, ShapeShiftType, ShiftType,
                    SliceType, FieldType, LenType, ReferenceType, TypeDescType>(
      ty);
}

bool verifySameLists(llvm::ArrayRef<RecordType::TypePair> a1,
                     llvm::ArrayRef<RecordType::TypePair> a2) {
  // FIXME: do we need to allow for any variance here?
  return a1 == a2;
}

static llvm::StringRef getVolatileKeyword() { return "volatile"; }

static aiir::ParseResult parseOptionalCommaAndKeyword(aiir::AsmParser &parser,
                                                      aiir::StringRef keyword,
                                                      bool &parsedKeyword) {
  if (!parser.parseOptionalComma()) {
    if (parser.parseKeyword(keyword))
      return aiir::failure();
    parsedKeyword = true;
    return aiir::success();
  }
  parsedKeyword = false;
  return aiir::success();
}

RecordType verifyDerived(aiir::AsmParser &parser, RecordType derivedTy,
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

aiir::Type fir::parseFirType(FIROpsDialect *dialect,
                             aiir::DialectAsmParser &parser) {
  aiir::StringRef typeTag;
  aiir::Type genType;
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
struct RecordTypeStorage : public aiir::TypeStorage {
  using KeyTy = llvm::StringRef;

  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.str());
  }

  bool operator==(const KeyTy &key) const { return key == getName(); }

  static RecordTypeStorage *construct(aiir::TypeStorageAllocator &allocator,
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

  bool isFinalized() const { return finalized; }
  void finalize(llvm::ArrayRef<RecordType::TypePair> lenParamList,
                llvm::ArrayRef<RecordType::TypePair> typeList) {
    if (finalized)
      return;
    finalized = true;
    setLenParamList(lenParamList);
    setTypeList(typeList);
  }

  bool isPacked() const { return packed; }
  void pack(bool p) { packed = p; }
  bool isSequence() const { return sequence; }
  void setSequence(bool s) { sequence = s; }

protected:
  std::string name;
  bool finalized;
  bool packed;
  bool sequence;
  std::vector<RecordType::TypePair> lens;
  std::vector<RecordType::TypePair> types;

private:
  RecordTypeStorage() = delete;
  explicit RecordTypeStorage(llvm::StringRef name)
      : name{name}, finalized{false}, packed{false}, sequence{false} {}
};

} // namespace detail

template <typename A, typename B>
bool inbounds(A v, B lb, B ub) {
  return v >= lb && v < ub;
}

bool isa_fir_type(aiir::Type t) {
  return llvm::isa<FIROpsDialect>(t.getDialect());
}

bool isa_std_type(aiir::Type t) {
  return llvm::isa<aiir::BuiltinDialect>(t.getDialect());
}

bool isa_fir_or_std_type(aiir::Type t) {
  if (auto funcType = aiir::dyn_cast<aiir::FunctionType>(t))
    return llvm::all_of(funcType.getInputs(), isa_fir_or_std_type) &&
           llvm::all_of(funcType.getResults(), isa_fir_or_std_type);
  return isa_fir_type(t) || isa_std_type(t);
}

aiir::Type getDerivedType(aiir::Type ty) {
  return llvm::TypeSwitch<aiir::Type, aiir::Type>(ty)
      .Case<fir::PointerType, fir::HeapType, fir::SequenceType>([](auto p) {
        if (auto seq = aiir::dyn_cast<fir::SequenceType>(p.getEleTy()))
          return seq.getEleTy();
        return p.getEleTy();
      })
      .Case([](fir::BaseBoxType p) { return getDerivedType(p.getEleTy()); })
      .Default([](aiir::Type t) { return t; });
}

aiir::Type updateTypeWithVolatility(aiir::Type type, bool isVolatile) {
  // If we already have the volatility we asked for, return the type unchanged.
  if (fir::isa_volatile_type(type) == isVolatile)
    return type;
  return aiir::TypeSwitch<aiir::Type, aiir::Type>(type)
      .Case<fir::BoxType, fir::ClassType, fir::ReferenceType>(
          [&](auto ty) -> aiir::Type {
            using TYPE = decltype(ty);
            return TYPE::get(ty.getEleTy(), isVolatile);
          })
      .Default([&](aiir::Type t) -> aiir::Type { return t; });
}

aiir::Type dyn_cast_ptrEleTy(aiir::Type t) {
  return llvm::TypeSwitch<aiir::Type, aiir::Type>(t)
      .Case<fir::ReferenceType, fir::PointerType, fir::HeapType,
            fir::LLVMPointerType>([](auto p) { return p.getEleTy(); })
      .Default([](aiir::Type) { return aiir::Type{}; });
}

aiir::Type dyn_cast_ptrOrBoxEleTy(aiir::Type t) {
  return llvm::TypeSwitch<aiir::Type, aiir::Type>(t)
      .Case<fir::ReferenceType, fir::PointerType, fir::HeapType,
            fir::LLVMPointerType>([](auto p) { return p.getEleTy(); })
      .Case<fir::BaseBoxType, fir::BoxCharType>(
          [](auto p) { return unwrapRefType(p.getEleTy()); })
      .Default([](aiir::Type) { return aiir::Type{}; });
}

static bool hasDynamicSize(fir::RecordType recTy) {
  if (recTy.getLenParamList().empty())
    return false;
  for (auto field : recTy.getTypeList()) {
    if (auto arr = aiir::dyn_cast<fir::SequenceType>(field.second)) {
      if (sequenceWithNonConstantShape(arr))
        return true;
    } else if (characterWithDynamicLen(field.second)) {
      return true;
    } else if (auto rec = aiir::dyn_cast<fir::RecordType>(field.second)) {
      if (hasDynamicSize(rec))
        return true;
    }
  }
  return false;
}

bool hasDynamicSize(aiir::Type t) {
  if (auto arr = aiir::dyn_cast<fir::SequenceType>(t)) {
    if (sequenceWithNonConstantShape(arr))
      return true;
    t = arr.getEleTy();
  }
  if (characterWithDynamicLen(t))
    return true;
  if (auto rec = aiir::dyn_cast<fir::RecordType>(t))
    return hasDynamicSize(rec);
  return false;
}

aiir::Type extractSequenceType(aiir::Type ty) {
  if (aiir::isa<fir::SequenceType>(ty))
    return ty;
  if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(ty))
    return extractSequenceType(boxTy.getEleTy());
  if (auto heapTy = aiir::dyn_cast<fir::HeapType>(ty))
    return extractSequenceType(heapTy.getEleTy());
  if (auto ptrTy = aiir::dyn_cast<fir::PointerType>(ty))
    return extractSequenceType(ptrTy.getEleTy());
  return aiir::Type{};
}

bool isPointerType(aiir::Type ty) {
  if (auto refTy = fir::dyn_cast_ptrEleTy(ty))
    ty = refTy;
  if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(ty))
    return aiir::isa<fir::PointerType>(boxTy.getEleTy());
  return false;
}

bool isAllocatableType(aiir::Type ty) {
  if (auto refTy = fir::dyn_cast_ptrEleTy(ty))
    ty = refTy;
  if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(ty))
    return aiir::isa<fir::HeapType>(boxTy.getEleTy());
  return false;
}

bool isBoxNone(aiir::Type ty) {
  if (auto box = aiir::dyn_cast<fir::BoxType>(ty))
    return aiir::isa<aiir::NoneType>(box.getEleTy());
  return false;
}

bool isBoxedRecordType(aiir::Type ty) {
  if (auto refTy = fir::dyn_cast_ptrEleTy(ty))
    ty = refTy;
  if (auto boxTy = aiir::dyn_cast<fir::BoxType>(ty)) {
    if (aiir::isa<fir::RecordType>(boxTy.getEleTy()))
      return true;
    aiir::Type innerType = boxTy.unwrapInnerType();
    return innerType && aiir::isa<fir::RecordType>(innerType);
  }
  return false;
}

// CLASS(*)
bool isClassStarType(aiir::Type ty) {
  if (auto clTy = aiir::dyn_cast<fir::ClassType>(fir::unwrapRefType(ty))) {
    if (aiir::isa<aiir::NoneType>(clTy.getEleTy()))
      return true;
    aiir::Type innerType = clTy.unwrapInnerType();
    return innerType && aiir::isa<aiir::NoneType>(innerType);
  }
  return false;
}

bool isScalarBoxedRecordType(aiir::Type ty) {
  if (auto refTy = fir::dyn_cast_ptrEleTy(ty))
    ty = refTy;
  if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(ty)) {
    if (aiir::isa<fir::RecordType>(boxTy.getEleTy()))
      return true;
    if (auto heapTy = aiir::dyn_cast<fir::HeapType>(boxTy.getEleTy()))
      return aiir::isa<fir::RecordType>(heapTy.getEleTy());
    if (auto ptrTy = aiir::dyn_cast<fir::PointerType>(boxTy.getEleTy()))
      return aiir::isa<fir::RecordType>(ptrTy.getEleTy());
  }
  return false;
}

bool isAssumedType(aiir::Type ty) {
  // Rule out CLASS(*) which are `fir.class<[fir.array] none>`.
  if (aiir::isa<fir::ClassType>(ty))
    return false;
  aiir::Type valueType = fir::unwrapPassByRefType(fir::unwrapRefType(ty));
  // Refuse raw `none` or `fir.array<none>` since assumed type
  // should be in memory variables.
  if (valueType == ty)
    return false;
  aiir::Type inner = fir::unwrapSequenceType(valueType);
  return aiir::isa<aiir::NoneType>(inner);
}

bool isAssumedShape(aiir::Type ty) {
  if (auto boxTy = aiir::dyn_cast<fir::BoxType>(ty))
    if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(boxTy.getEleTy()))
      return seqTy.hasDynamicExtents();
  return false;
}

bool isAllocatableOrPointerArray(aiir::Type ty) {
  if (auto refTy = fir::dyn_cast_ptrEleTy(ty))
    ty = refTy;
  if (auto boxTy = aiir::dyn_cast<fir::BoxType>(ty)) {
    if (auto heapTy = aiir::dyn_cast<fir::HeapType>(boxTy.getEleTy()))
      return aiir::isa<fir::SequenceType>(heapTy.getEleTy());
    if (auto ptrTy = aiir::dyn_cast<fir::PointerType>(boxTy.getEleTy()))
      return aiir::isa<fir::SequenceType>(ptrTy.getEleTy());
  }
  return false;
}

bool isTypeWithDescriptor(aiir::Type ty) {
  if (aiir::isa<fir::BaseBoxType>(unwrapRefType(ty)))
    return true;
  return false;
}

bool isPolymorphicType(aiir::Type ty) {
  // CLASS(T) or CLASS(*)
  if (aiir::isa<fir::ClassType>(fir::unwrapRefType(ty)))
    return true;
  // assumed type are polymorphic.
  return isAssumedType(ty);
}

bool isUnlimitedPolymorphicType(aiir::Type ty) {
  // CLASS(*)
  if (isClassStarType(ty))
    return true;
  // TYPE(*)
  return isAssumedType(ty);
}

aiir::Type unwrapInnerType(aiir::Type ty) {
  return llvm::TypeSwitch<aiir::Type, aiir::Type>(ty)
      .Case<fir::PointerType, fir::HeapType, fir::SequenceType>([](auto t) {
        aiir::Type eleTy = t.getEleTy();
        if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(eleTy))
          return seqTy.getEleTy();
        return eleTy;
      })
      .Case([](fir::RecordType t) { return t; })
      .Default([](aiir::Type) { return aiir::Type{}; });
}

bool isRecordWithAllocatableMember(aiir::Type ty) {
  if (auto recTy = aiir::dyn_cast<fir::RecordType>(ty))
    for (auto [field, memTy] : recTy.getTypeList()) {
      if (fir::isAllocatableType(memTy))
        return true;
      // A record type cannot recursively include itself as a direct member.
      // There must be an intervening `ptr` type, so recursion is safe here.
      if (aiir::isa<fir::RecordType>(memTy) &&
          isRecordWithAllocatableMember(memTy))
        return true;
    }
  return false;
}

bool isRecordWithDescriptorMember(aiir::Type ty) {
  ty = unwrapSequenceType(ty);
  if (auto recTy = aiir::dyn_cast<fir::RecordType>(ty))
    for (auto [field, memTy] : recTy.getTypeList()) {
      memTy = unwrapSequenceType(memTy);
      if (aiir::isa<fir::BaseBoxType>(memTy))
        return true;
      if (aiir::isa<fir::RecordType>(memTy) &&
          isRecordWithDescriptorMember(memTy))
        return true;
    }
  return false;
}

aiir::Type unwrapAllRefAndSeqType(aiir::Type ty) {
  while (true) {
    aiir::Type nt = unwrapSequenceType(unwrapRefType(ty));
    if (auto vecTy = aiir::dyn_cast<fir::VectorType>(nt))
      nt = vecTy.getEleTy();
    if (nt == ty)
      return ty;
    ty = nt;
  }
}

aiir::Type getFortranElementType(aiir::Type ty) {
  return fir::unwrapSequenceType(
      fir::unwrapPassByRefType(fir::unwrapRefType(ty)));
}

aiir::Type unwrapSeqOrBoxedSeqType(aiir::Type ty) {
  if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(ty))
    return seqTy.getEleTy();
  if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(ty)) {
    auto eleTy = unwrapRefType(boxTy.getEleTy());
    if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(eleTy))
      return seqTy.getEleTy();
  }
  return ty;
}

unsigned getBoxRank(aiir::Type boxTy) {
  auto eleTy = fir::dyn_cast_ptrOrBoxEleTy(boxTy);
  if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(eleTy))
    return seqTy.getDimension();
  return 0;
}

/// Return the ISO_C_BINDING intrinsic module value of type \p ty.
int getTypeCode(aiir::Type ty, const fir::KindMapping &kindMap) {
  if (aiir::IntegerType intTy = aiir::dyn_cast<aiir::IntegerType>(ty)) {
    if (intTy.isUnsigned()) {
      switch (intTy.getWidth()) {
      case 8:
        return CFI_type_uint8_t;
      case 16:
        return CFI_type_uint16_t;
      case 32:
        return CFI_type_uint32_t;
      case 64:
        return CFI_type_uint64_t;
      case 128:
        return CFI_type_uint128_t;
      }
      llvm_unreachable("unsupported integer type");
    } else {
      switch (intTy.getWidth()) {
      case 8:
        return CFI_type_int8_t;
      case 16:
        return CFI_type_int16_t;
      case 32:
        return CFI_type_int32_t;
      case 64:
        return CFI_type_int64_t;
      case 128:
        return CFI_type_int128_t;
      }
      llvm_unreachable("unsupported integer type");
    }
  }
  if (fir::LogicalType logicalTy = aiir::dyn_cast<fir::LogicalType>(ty)) {
    switch (kindMap.getLogicalBitsize(logicalTy.getFKind())) {
    case 8:
      return CFI_type_Bool;
    case 16:
      return CFI_type_int_least16_t;
    case 32:
      return CFI_type_int_least32_t;
    case 64:
      return CFI_type_int_least64_t;
    }
    llvm_unreachable("unsupported logical type");
  }
  if (aiir::FloatType floatTy = aiir::dyn_cast<aiir::FloatType>(ty)) {
    switch (floatTy.getWidth()) {
    case 16:
      return floatTy.isBF16() ? CFI_type_bfloat : CFI_type_half_float;
    case 32:
      return CFI_type_float;
    case 64:
      return CFI_type_double;
    case 80:
      return CFI_type_extended_double;
    case 128:
      return CFI_type_float128;
    }
    llvm_unreachable("unsupported real type");
  }
  if (aiir::ComplexType complexTy = aiir::dyn_cast<aiir::ComplexType>(ty)) {
    aiir::FloatType floatTy =
        aiir::cast<aiir::FloatType>(complexTy.getElementType());
    if (floatTy.isBF16())
      return CFI_type_bfloat_Complex;
    switch (floatTy.getWidth()) {
    case 16:
      return CFI_type_half_float_Complex;
    case 32:
      return CFI_type_float_Complex;
    case 64:
      return CFI_type_double_Complex;
    case 80:
      return CFI_type_extended_double_Complex;
    case 128:
      return CFI_type_float128_Complex;
    }
    llvm_unreachable("unsupported complex size");
  }
  if (fir::CharacterType charTy = aiir::dyn_cast<fir::CharacterType>(ty)) {
    switch (kindMap.getCharacterBitsize(charTy.getFKind())) {
    case 8:
      return CFI_type_char;
    case 16:
      return CFI_type_char16_t;
    case 32:
      return CFI_type_char32_t;
    }
    llvm_unreachable("unsupported character type");
  }
  if (fir::isa_ref_type(ty))
    return CFI_type_cptr;
  if (aiir::isa<fir::RecordType>(ty))
    return CFI_type_struct;
  llvm_unreachable("unsupported type");
}

std::string getTypeAsString(aiir::Type ty, const fir::KindMapping &kindMap,
                            llvm::StringRef prefix) {
  std::string buf = prefix.str();
  llvm::raw_string_ostream name{buf};
  if (!prefix.empty())
    name << "_";

  std::function<void(aiir::Type)> appendTypeName = [&](aiir::Type ty) {
    while (ty) {
      if (fir::isa_trivial(ty)) {
        if (aiir::isa<aiir::IndexType>(ty)) {
          name << "idx";
        } else if (ty.isIntOrIndex()) {
          name << 'i' << ty.getIntOrFloatBitWidth();
        } else if (aiir::isa<aiir::FloatType>(ty)) {
          name << 'f' << ty.getIntOrFloatBitWidth();
        } else if (auto cplxTy =
                       aiir::dyn_cast_or_null<aiir::ComplexType>(ty)) {
          name << 'z';
          auto floatTy = aiir::cast<aiir::FloatType>(cplxTy.getElementType());
          name << floatTy.getWidth();
        } else if (auto logTy = aiir::dyn_cast_or_null<fir::LogicalType>(ty)) {
          name << 'l' << kindMap.getLogicalBitsize(logTy.getFKind());
        } else {
          llvm::report_fatal_error("unsupported type");
        }
        break;
      } else if (aiir::isa<aiir::NoneType>(ty)) {
        name << "none";
        break;
      } else if (auto charTy = aiir::dyn_cast_or_null<fir::CharacterType>(ty)) {
        name << 'c' << kindMap.getCharacterBitsize(charTy.getFKind());
        if (charTy.getLen() == fir::CharacterType::unknownLen())
          name << "xU";
        else if (charTy.getLen() != fir::CharacterType::singleton())
          name << "x" << charTy.getLen();
        break;
      } else if (auto seqTy = aiir::dyn_cast_or_null<fir::SequenceType>(ty)) {
        for (auto extent : seqTy.getShape()) {
          if (extent == fir::SequenceType::getUnknownExtent())
            name << "Ux";
          else
            name << extent << 'x';
        }
        ty = seqTy.getEleTy();
      } else if (auto refTy = aiir::dyn_cast_or_null<fir::ReferenceType>(ty)) {
        name << "ref_";
        ty = refTy.getEleTy();
      } else if (auto ptrTy = aiir::dyn_cast_or_null<fir::PointerType>(ty)) {
        name << "ptr_";
        ty = ptrTy.getEleTy();
      } else if (auto ptrTy =
                     aiir::dyn_cast_or_null<fir::LLVMPointerType>(ty)) {
        name << "llvmptr_";
        ty = ptrTy.getEleTy();
      } else if (auto heapTy = aiir::dyn_cast_or_null<fir::HeapType>(ty)) {
        name << "heap_";
        ty = heapTy.getEleTy();
      } else if (auto classTy = aiir::dyn_cast_or_null<fir::ClassType>(ty)) {
        name << "class_";
        ty = classTy.getEleTy();
      } else if (auto boxTy = aiir::dyn_cast_or_null<fir::BoxType>(ty)) {
        name << "box_";
        ty = boxTy.getEleTy();
      } else if (auto boxcharTy =
                     aiir::dyn_cast_or_null<fir::BoxCharType>(ty)) {
        name << "boxchar_";
        ty = boxcharTy.getEleTy();
      } else if (auto boxprocTy =
                     aiir::dyn_cast_or_null<fir::BoxProcType>(ty)) {
        name << "boxproc_";
        auto procTy = aiir::dyn_cast<aiir::FunctionType>(boxprocTy.getEleTy());
        assert(procTy.getNumResults() <= 1 &&
               "function type with more than one result");
        for (const auto &result : procTy.getResults())
          appendTypeName(result);
        name << "_args";
        for (const auto &arg : procTy.getInputs()) {
          name << '_';
          appendTypeName(arg);
        }
        break;
      } else if (auto recTy = aiir::dyn_cast_or_null<fir::RecordType>(ty)) {
        name << "rec_" << recTy.getName();
        break;
      } else {
        llvm::report_fatal_error("unsupported type");
      }
    }
  };

  appendTypeName(ty);
  return buf;
}

static aiir::Type changeElementTypeImpl(aiir::Type type,
                                        aiir::Type newElementType,
                                        bool turnBoxIntoClass,
                                        bool turnClassIntoBox) {
  return llvm::TypeSwitch<aiir::Type, aiir::Type>(type)
      .Case([&](fir::SequenceType seqTy) -> aiir::Type {
        return fir::SequenceType::get(seqTy.getShape(), newElementType);
      })
      .Case<fir::ReferenceType>([&](auto t) -> aiir::Type {
        using FIRT = decltype(t);
        auto newEleTy = changeElementTypeImpl(
            t.getEleTy(), newElementType, turnBoxIntoClass, turnClassIntoBox);
        return FIRT::get(newEleTy, t.isVolatile());
      })
      .Case<fir::PointerType, fir::HeapType>([&](auto t) -> aiir::Type {
        using FIRT = decltype(t);
        return FIRT::get(changeElementTypeImpl(
            t.getEleTy(), newElementType, turnBoxIntoClass, turnClassIntoBox));
      })
      .Case([&](fir::BoxType t) -> aiir::Type {
        aiir::Type newInnerType =
            changeElementTypeImpl(t.getEleTy(), newElementType, false, false);
        if (turnBoxIntoClass)
          return fir::ClassType::get(newInnerType, t.isVolatile());
        return fir::BoxType::get(newInnerType, t.isVolatile());
      })
      .Case([&](fir::ClassType t) -> aiir::Type {
        aiir::Type newInnerType =
            changeElementTypeImpl(t.getEleTy(), newElementType, false, false);
        if (turnClassIntoBox)
          return fir::BoxType::get(newInnerType, t.isVolatile());
        return fir::ClassType::get(newInnerType, t.isVolatile());
      })
      .Default([&](aiir::Type t) -> aiir::Type {
        assert((fir::isa_trivial(t) || llvm::isa<fir::RecordType>(t) ||
                llvm::isa<aiir::NoneType>(t)) &&
               "unexpected FIR leaf type");
        return newElementType;
      });
}

aiir::Type changeElementType(aiir::Type type, aiir::Type newElementType,
                             bool turnBoxIntoClass) {
  return changeElementTypeImpl(type, newElementType, turnBoxIntoClass,
                               /*turnClassIntoBox=*/false);
}

} // namespace fir

namespace {

static llvm::SmallPtrSet<detail::RecordTypeStorage const *, 4>
    recordTypeVisited;

} // namespace

void fir::verifyIntegralType(aiir::Type type) {
  if (isaIntegerType(type) || aiir::isa<aiir::IndexType>(type))
    return;
  llvm::report_fatal_error("expected integral type");
}

void fir::printFirType(FIROpsDialect *, aiir::Type ty,
                       aiir::DialectAsmPrinter &p) {
  if (aiir::failed(generatedTypePrinter(ty, p)))
    llvm::report_fatal_error("unknown type to print");
}

bool fir::isa_unknown_size_box(aiir::Type t) {
  if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(t)) {
    auto valueType = fir::unwrapPassByRefType(boxTy);
    if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(valueType))
      if (seqTy.hasUnknownShape())
        return true;
  }
  return false;
}

bool fir::isa_volatile_type(aiir::Type t) {
  return llvm::TypeSwitch<aiir::Type, bool>(t)
      .Case<fir::ReferenceType, fir::BoxType, fir::ClassType>(
          [](auto t) { return t.isVolatile(); })
      .Default([](aiir::Type) { return false; });
}

//===----------------------------------------------------------------------===//
// BoxProcType
//===----------------------------------------------------------------------===//

// `boxproc` `<` return-type `>`
aiir::Type BoxProcType::parse(aiir::AsmParser &parser) {
  aiir::Type ty;
  if (parser.parseLess() || parser.parseType(ty) || parser.parseGreater())
    return {};
  return get(parser.getContext(), ty);
}

void fir::BoxProcType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getEleTy() << '>';
}

llvm::LogicalResult
BoxProcType::verify(llvm::function_ref<aiir::InFlightDiagnostic()> emitError,
                    aiir::Type eleTy) {
  if (aiir::isa<aiir::FunctionType>(eleTy))
    return aiir::success();
  if (auto refTy = aiir::dyn_cast<ReferenceType>(eleTy))
    if (aiir::isa<aiir::FunctionType>(refTy))
      return aiir::success();
  return emitError() << "invalid type for boxproc" << eleTy << '\n';
}

static bool cannotBePointerOrHeapElementType(aiir::Type eleTy) {
  return aiir::isa<BoxType, BoxCharType, BoxProcType, ShapeType, ShapeShiftType,
                   SliceType, FieldType, LenType, HeapType, PointerType,
                   ReferenceType, TypeDescType>(eleTy);
}

//===----------------------------------------------------------------------===//
// BoxType
//===----------------------------------------------------------------------===//

// `box` `<` type (`, volatile` $volatile^)? `>`
aiir::Type fir::BoxType::parse(aiir::AsmParser &parser) {
  aiir::Type eleTy;
  auto location = parser.getCurrentLocation();
  auto *context = parser.getContext();
  bool isVolatile = false;
  if (parser.parseLess() || parser.parseType(eleTy))
    return {};
  if (parseOptionalCommaAndKeyword(parser, getVolatileKeyword(), isVolatile))
    return {};
  if (parser.parseGreater())
    return {};
  return parser.getChecked<fir::BoxType>(location, context, eleTy, isVolatile);
}

void fir::BoxType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getEleTy();
  if (isVolatile())
    printer << ", " << getVolatileKeyword();
  printer << '>';
}

llvm::LogicalResult
fir::BoxType::verify(llvm::function_ref<aiir::InFlightDiagnostic()> emitError,
                     aiir::Type eleTy, bool isVolatile) {
  if (aiir::isa<fir::BaseBoxType>(eleTy))
    return emitError() << "invalid element type\n";
  // TODO
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// BoxCharType
//===----------------------------------------------------------------------===//

aiir::Type fir::BoxCharType::parse(aiir::AsmParser &parser) {
  return parseKindSingleton<fir::BoxCharType>(parser);
}

void fir::BoxCharType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getKind() << ">";
}

CharacterType
fir::BoxCharType::getElementType(aiir::AIIRContext *context) const {
  return CharacterType::getUnknownLen(context, getKind());
}

CharacterType fir::BoxCharType::getEleTy() const {
  return getElementType(getContext());
}

//===----------------------------------------------------------------------===//
// CharacterType
//===----------------------------------------------------------------------===//

// `char` `<` kind [`,` `len`] `>`
aiir::Type fir::CharacterType::parse(aiir::AsmParser &parser) {
  int kind = 0;
  if (parser.parseLess() || parser.parseInteger(kind))
    return {};
  CharacterType::LenType len = 1;
  if (aiir::succeeded(parser.parseOptionalComma())) {
    if (aiir::succeeded(parser.parseOptionalQuestion())) {
      len = fir::CharacterType::unknownLen();
    } else if (!aiir::succeeded(parser.parseInteger(len))) {
      return {};
    }
  }
  if (parser.parseGreater())
    return {};
  return get(parser.getContext(), kind, len);
}

void fir::CharacterType::print(aiir::AsmPrinter &printer) const {
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

// `class` `<` type (`, volatile` $volatile^)? `>`
aiir::Type fir::ClassType::parse(aiir::AsmParser &parser) {
  aiir::Type eleTy;
  auto location = parser.getCurrentLocation();
  auto *context = parser.getContext();
  bool isVolatile = false;
  if (parser.parseLess() || parser.parseType(eleTy))
    return {};
  if (parseOptionalCommaAndKeyword(parser, getVolatileKeyword(), isVolatile))
    return {};
  if (parser.parseGreater())
    return {};
  return parser.getChecked<fir::ClassType>(location, context, eleTy,
                                           isVolatile);
}

void fir::ClassType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getEleTy();
  if (isVolatile())
    printer << ", " << getVolatileKeyword();
  printer << '>';
}

llvm::LogicalResult
fir::ClassType::verify(llvm::function_ref<aiir::InFlightDiagnostic()> emitError,
                       aiir::Type eleTy, bool isVolatile) {
  if (aiir::isa<fir::RecordType, fir::SequenceType, fir::HeapType,
                fir::PointerType, aiir::NoneType, aiir::IntegerType,
                aiir::FloatType, fir::CharacterType, fir::LogicalType,
                aiir::ComplexType>(eleTy))
    return aiir::success();
  return emitError() << "invalid element type\n";
}

//===----------------------------------------------------------------------===//
// HeapType
//===----------------------------------------------------------------------===//

// `heap` `<` type `>`
aiir::Type fir::HeapType::parse(aiir::AsmParser &parser) {
  return parseTypeSingleton<HeapType>(parser);
}

void fir::HeapType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getEleTy() << '>';
}

llvm::LogicalResult
fir::HeapType::verify(llvm::function_ref<aiir::InFlightDiagnostic()> emitError,
                      aiir::Type eleTy) {
  if (cannotBePointerOrHeapElementType(eleTy))
    return emitError() << "cannot build a heap pointer to type: " << eleTy
                       << '\n';
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// IntegerType
//===----------------------------------------------------------------------===//

// `int` `<` kind `>`
aiir::Type fir::IntegerType::parse(aiir::AsmParser &parser) {
  return parseKindSingleton<fir::IntegerType>(parser);
}

void fir::IntegerType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getFKind() << '>';
}

//===----------------------------------------------------------------------===//
// UnsignedType
//===----------------------------------------------------------------------===//

// `unsigned` `<` kind `>`
aiir::Type fir::UnsignedType::parse(aiir::AsmParser &parser) {
  return parseKindSingleton<fir::UnsignedType>(parser);
}

void fir::UnsignedType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getFKind() << '>';
}

//===----------------------------------------------------------------------===//
// LogicalType
//===----------------------------------------------------------------------===//

// `logical` `<` kind `>`
aiir::Type fir::LogicalType::parse(aiir::AsmParser &parser) {
  return parseKindSingleton<fir::LogicalType>(parser);
}

void fir::LogicalType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getFKind() << '>';
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

// `ptr` `<` type `>`
aiir::Type fir::PointerType::parse(aiir::AsmParser &parser) {
  return parseTypeSingleton<fir::PointerType>(parser);
}

void fir::PointerType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getEleTy() << '>';
}

llvm::LogicalResult fir::PointerType::verify(
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError,
    aiir::Type eleTy) {
  if (cannotBePointerOrHeapElementType(eleTy))
    return emitError() << "cannot build a pointer to type: " << eleTy << '\n';
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// RecordType
//===----------------------------------------------------------------------===//

// Fortran derived type
// unpacked:
// `type` `<` name
//           (`(` id `:` type (`,` id `:` type)* `)`)?
//           (`{` id `:` type (`,` id `:` type)* `}`)? '>'
// packed:
// `type` `<` name
//           (`(` id `:` type (`,` id `:` type)* `)`)?
//           (`<{` id `:` type (`,` id `:` type)* `}>`)? '>'
aiir::Type fir::RecordType::parse(aiir::AsmParser &parser) {
  llvm::StringRef name;
  if (parser.parseLess() || parser.parseKeyword(&name))
    return {};
  RecordType result = RecordType::get(parser.getContext(), name);
  // Optional SEQUENCE attribute: ", sequence"
  if (!parser.parseOptionalComma()) {
    if (parser.parseKeyword("sequence")) {
      parser.emitError(parser.getNameLoc(), "expected 'sequence' keyword");
      return {};
    }
    result.setSequence(true);
  }

  RecordType::TypeVector lenParamList;
  if (!parser.parseOptionalLParen()) {
    while (true) {
      llvm::StringRef lenparam;
      aiir::Type intTy;
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

  RecordType::TypeVector typeList;
  if (!parser.parseOptionalLess()) {
    result.pack(true);
  }

  if (!parser.parseOptionalLBrace()) {
    while (true) {
      llvm::StringRef field;
      aiir::Type fldTy;
      if (parser.parseKeyword(&field) || parser.parseColon() ||
          parser.parseType(fldTy)) {
        parser.emitError(parser.getNameLoc(), "expected field type list");
        return {};
      }
      typeList.emplace_back(field, fldTy);
      if (parser.parseOptionalComma())
        break;
    }
    if (parser.parseOptionalGreater()) {
      if (parser.parseRBrace())
        return {};
    }
  }

  if (parser.parseGreater())
    return {};

  if (lenParamList.empty() && typeList.empty())
    return result;

  result.finalize(lenParamList, typeList);
  return verifyDerived(parser, result, lenParamList, typeList);
}

void fir::RecordType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getName();
  if (isSequence())
    printer << ",sequence";
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
      if (isPacked()) {
        printer << '<';
      }
      char ch = '{';
      for (auto p : getTypeList()) {
        printer << ch << p.first << ':';
        p.second.print(printer.getStream());
        ch = ',';
      }
      printer << '}';
      if (isPacked()) {
        printer << '>';
      }
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

bool fir::RecordType::isFinalized() const { return getImpl()->isFinalized(); }

void fir::RecordType::pack(bool p) { getImpl()->pack(p); }

bool fir::RecordType::isPacked() const { return getImpl()->isPacked(); }

bool fir::RecordType::isSequence() const { return getImpl()->isSequence(); }

void fir::RecordType::setSequence(bool s) { getImpl()->setSequence(s); }

detail::RecordTypeStorage const *fir::RecordType::uniqueKey() const {
  return getImpl();
}

llvm::LogicalResult fir::RecordType::verify(
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError,
    llvm::StringRef name) {
  if (name.size() == 0)
    return emitError() << "record types must have a name";
  return aiir::success();
}

aiir::Type fir::RecordType::getType(llvm::StringRef ident) {
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

// `ref` `<` type (`, volatile` $volatile^)? `>`
aiir::Type fir::ReferenceType::parse(aiir::AsmParser &parser) {
  auto location = parser.getCurrentLocation();
  auto *context = parser.getContext();
  aiir::Type eleTy;
  bool isVolatile = false;
  if (parser.parseLess() || parser.parseType(eleTy))
    return {};
  if (parseOptionalCommaAndKeyword(parser, getVolatileKeyword(), isVolatile))
    return {};
  if (parser.parseGreater())
    return {};
  return parser.getChecked<fir::ReferenceType>(location, context, eleTy,
                                               isVolatile);
}

void fir::ReferenceType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getEleTy();
  if (isVolatile())
    printer << ", " << getVolatileKeyword();
  printer << '>';
}

llvm::LogicalResult fir::ReferenceType::verify(
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError, aiir::Type eleTy,
    bool isVolatile) {
  if (aiir::isa<ShapeType, ShapeShiftType, SliceType, FieldType, LenType,
                ReferenceType, TypeDescType>(eleTy))
    return emitError() << "cannot build a reference to type: " << eleTy << '\n';
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// SequenceType
//===----------------------------------------------------------------------===//

// `array` `<` `*` | bounds (`x` bounds)* `:` type (',' affine-map)? `>`
// bounds ::= `?` | int-lit
aiir::Type fir::SequenceType::parse(aiir::AsmParser &parser) {
  if (parser.parseLess())
    return {};
  SequenceType::Shape shape;
  if (parser.parseOptionalStar()) {
    if (parser.parseDimensionList(shape, /*allowDynamic=*/true))
      return {};
  } else if (parser.parseColon()) {
    return {};
  }
  aiir::Type eleTy;
  if (parser.parseType(eleTy))
    return {};
  aiir::AffineMapAttr map;
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

void fir::SequenceType::print(aiir::AsmPrinter &printer) const {
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

llvm::LogicalResult fir::SequenceType::verify(
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, aiir::Type eleTy,
    aiir::AffineMapAttr layoutMap) {
  // DIMENSION attribute can only be applied to an intrinsic or record type
  if (aiir::isa<BoxType, BoxCharType, BoxProcType, ShapeType, ShapeShiftType,
                ShiftType, SliceType, FieldType, LenType, HeapType, PointerType,
                ReferenceType, TypeDescType, SequenceType>(eleTy))
    return emitError() << "cannot build an array of this element type: "
                       << eleTy << '\n';
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// ShapeType
//===----------------------------------------------------------------------===//

aiir::Type fir::ShapeType::parse(aiir::AsmParser &parser) {
  return parseRankSingleton<fir::ShapeType>(parser);
}

void fir::ShapeType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getImpl()->rank << ">";
}

//===----------------------------------------------------------------------===//
// ShapeShiftType
//===----------------------------------------------------------------------===//

aiir::Type fir::ShapeShiftType::parse(aiir::AsmParser &parser) {
  return parseRankSingleton<fir::ShapeShiftType>(parser);
}

void fir::ShapeShiftType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getRank() << ">";
}

//===----------------------------------------------------------------------===//
// ShiftType
//===----------------------------------------------------------------------===//

aiir::Type fir::ShiftType::parse(aiir::AsmParser &parser) {
  return parseRankSingleton<fir::ShiftType>(parser);
}

void fir::ShiftType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getRank() << ">";
}

//===----------------------------------------------------------------------===//
// SliceType
//===----------------------------------------------------------------------===//

// `slice` `<` rank `>`
aiir::Type fir::SliceType::parse(aiir::AsmParser &parser) {
  return parseRankSingleton<fir::SliceType>(parser);
}

void fir::SliceType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getRank() << '>';
}

//===----------------------------------------------------------------------===//
// TypeDescType
//===----------------------------------------------------------------------===//

// `tdesc` `<` type `>`
aiir::Type fir::TypeDescType::parse(aiir::AsmParser &parser) {
  return parseTypeSingleton<fir::TypeDescType>(parser);
}

void fir::TypeDescType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getOfTy() << '>';
}

llvm::LogicalResult fir::TypeDescType::verify(
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError,
    aiir::Type eleTy) {
  if (aiir::isa<BoxType, BoxCharType, BoxProcType, ShapeType, ShapeShiftType,
                ShiftType, SliceType, FieldType, LenType, ReferenceType,
                TypeDescType>(eleTy))
    return emitError() << "cannot build a type descriptor of type: " << eleTy
                       << '\n';
  return aiir::success();
}

//===----------------------------------------------------------------------===//
// VectorType
//===----------------------------------------------------------------------===//

// `vector` `<` len `:` type `>`
aiir::Type fir::VectorType::parse(aiir::AsmParser &parser) {
  int64_t len = 0;
  aiir::Type eleTy;
  if (parser.parseLess() || parser.parseInteger(len) || parser.parseColon() ||
      parser.parseType(eleTy) || parser.parseGreater())
    return {};
  return fir::VectorType::get(len, eleTy);
}

void fir::VectorType::print(aiir::AsmPrinter &printer) const {
  printer << "<" << getLen() << ':' << getEleTy() << '>';
}

llvm::LogicalResult fir::VectorType::verify(
    llvm::function_ref<aiir::InFlightDiagnostic()> emitError, uint64_t len,
    aiir::Type eleTy) {
  if (!(fir::isa_real(eleTy) || fir::isa_integer(eleTy)))
    return emitError() << "cannot build a vector of type " << eleTy << '\n';
  return aiir::success();
}

bool fir::VectorType::isValidElementType(aiir::Type t) {
  return isa_real(t) || isa_integer(t);
}

bool fir::isCharacterProcedureTuple(aiir::Type ty, bool acceptRawFunc) {
  aiir::TupleType tuple = aiir::dyn_cast<aiir::TupleType>(ty);
  return tuple && tuple.size() == 2 &&
         (aiir::isa<fir::BoxProcType>(tuple.getType(0)) ||
          (acceptRawFunc && aiir::isa<aiir::FunctionType>(tuple.getType(0)))) &&
         fir::isa_integer(tuple.getType(1));
}

bool fir::hasAbstractResult(aiir::FunctionType ty) {
  if (ty.getNumResults() == 0)
    return false;
  auto resultType = ty.getResult(0);
  return aiir::isa<fir::SequenceType, fir::BaseBoxType, fir::RecordType>(
      resultType);
}

/// Convert llvm::Type::TypeID to aiir::Type. \p kind is provided for error
/// messages only.
aiir::Type fir::fromRealTypeID(aiir::AIIRContext *context,
                               llvm::Type::TypeID typeID, fir::KindTy kind) {
  switch (typeID) {
  case llvm::Type::TypeID::HalfTyID:
    return aiir::Float16Type::get(context);
  case llvm::Type::TypeID::BFloatTyID:
    return aiir::BFloat16Type::get(context);
  case llvm::Type::TypeID::FloatTyID:
    return aiir::Float32Type::get(context);
  case llvm::Type::TypeID::DoubleTyID:
    return aiir::Float64Type::get(context);
  case llvm::Type::TypeID::X86_FP80TyID:
    return aiir::Float80Type::get(context);
  case llvm::Type::TypeID::FP128TyID:
    return aiir::Float128Type::get(context);
  default:
    aiir::emitError(aiir::UnknownLoc::get(context))
        << "unsupported type: !fir.real<" << kind << ">";
    return {};
  }
}

//===----------------------------------------------------------------------===//
// BaseBoxType
//===----------------------------------------------------------------------===//

aiir::Type BaseBoxType::getEleTy() const {
  return llvm::TypeSwitch<fir::BaseBoxType, aiir::Type>(*this)
      .Case<fir::BoxType, fir::ClassType>(
          [](auto type) { return type.getEleTy(); });
}

aiir::Type BaseBoxType::getBaseAddressType(bool dropHeapOrPtr) const {
  aiir::Type eleTy = getEleTy();
  if (!dropHeapOrPtr && fir::isa_ref_type(eleTy))
    return eleTy;
  return fir::ReferenceType::get(getElementOrSequenceType(), isVolatile());
}

aiir::Type BaseBoxType::unwrapInnerType() const {
  return fir::unwrapInnerType(getEleTy());
}

aiir::Type BaseBoxType::getElementOrSequenceType() const {
  aiir::Type eleTy = getEleTy();
  if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(eleTy))
    return seqTy;
  return fir::unwrapRefType(eleTy);
}

static aiir::Type
changeTypeShape(aiir::Type type,
                std::optional<fir::SequenceType::ShapeRef> newShape) {
  return llvm::TypeSwitch<aiir::Type, aiir::Type>(type)
      .Case([&](fir::SequenceType seqTy) -> aiir::Type {
        if (newShape)
          return fir::SequenceType::get(*newShape, seqTy.getEleTy());
        return seqTy.getEleTy();
      })
      .Case<fir::ReferenceType, fir::BoxType, fir::ClassType>(
          [&](auto t) -> aiir::Type {
            using FIRT = decltype(t);
            return FIRT::get(changeTypeShape(t.getEleTy(), newShape),
                             t.isVolatile());
          })
      .Case<fir::PointerType, fir::HeapType>([&](auto t) -> aiir::Type {
        using FIRT = decltype(t);
        return FIRT::get(changeTypeShape(t.getEleTy(), newShape));
      })
      .Default([&](aiir::Type t) -> aiir::Type {
        assert((fir::isa_trivial(t) || llvm::isa<fir::RecordType>(t) ||
                llvm::isa<aiir::NoneType>(t) ||
                llvm::isa<fir::CharacterType>(t)) &&
               "unexpected FIR leaf type");
        if (newShape)
          return fir::SequenceType::get(*newShape, t);
        return t;
      });
}

fir::BaseBoxType
fir::BaseBoxType::getBoxTypeWithNewShape(aiir::Type shapeMold) const {
  fir::SequenceType seqTy = fir::unwrapUntilSeqType(shapeMold);
  std::optional<fir::SequenceType::ShapeRef> newShape;
  if (seqTy)
    newShape = seqTy.getShape();
  return aiir::cast<fir::BaseBoxType>(changeTypeShape(*this, newShape));
}

fir::BaseBoxType fir::BaseBoxType::getBoxTypeWithNewShape(int rank) const {
  std::optional<fir::SequenceType::ShapeRef> newShape;
  fir::SequenceType::Shape shapeVector;
  if (rank > 0) {
    shapeVector =
        fir::SequenceType::Shape(rank, fir::SequenceType::getUnknownExtent());
    newShape = shapeVector;
  }
  return aiir::cast<fir::BaseBoxType>(changeTypeShape(*this, newShape));
}

fir::BaseBoxType
fir::BaseBoxType::getBoxTypeWithNewElementType(aiir::Type elementType,
                                               bool polymorphic) const {
  return llvm::cast<fir::BaseBoxType>(changeElementTypeImpl(
      *this, elementType, /*turnBoxIntoClass=*/polymorphic,
      /*turnClassIntoBox=*/!polymorphic));
}

fir::BaseBoxType fir::BaseBoxType::getBoxTypeWithNewAttr(
    fir::BaseBoxType::Attribute attr) const {
  aiir::Type baseType = fir::unwrapRefType(getEleTy());
  switch (attr) {
  case fir::BaseBoxType::Attribute::None:
    break;
  case fir::BaseBoxType::Attribute::Allocatable:
    baseType = fir::HeapType::get(baseType);
    break;
  case fir::BaseBoxType::Attribute::Pointer:
    baseType = fir::PointerType::get(baseType);
    break;
  }
  return llvm::TypeSwitch<fir::BaseBoxType, fir::BaseBoxType>(*this)
      .Case([baseType](fir::BoxType b) {
        return fir::BoxType::get(baseType, b.isVolatile());
      })
      .Case([baseType](fir::ClassType b) {
        return fir::ClassType::get(baseType, b.isVolatile());
      });
}

bool fir::BaseBoxType::isAssumedRank() const {
  if (auto seqTy =
          aiir::dyn_cast<fir::SequenceType>(fir::unwrapRefType(getEleTy())))
    return seqTy.hasUnknownShape();
  return false;
}

bool fir::BaseBoxType::isPointer() const {
  return llvm::isa<fir::PointerType>(getEleTy());
}

bool fir::BaseBoxType::isPointerOrAllocatable() const {
  return llvm::isa<fir::PointerType, fir::HeapType>(getEleTy());
}

bool BaseBoxType::isVolatile() const { return fir::isa_volatile_type(*this); }

bool BaseBoxType::isArray() const {
  return llvm::isa<fir::SequenceType>(getElementOrSequenceType());
}

//===----------------------------------------------------------------------===//
// FIROpsDialect
//===----------------------------------------------------------------------===//

void FIROpsDialect::registerTypes() {
  addTypes<BoxType, BoxCharType, BoxProcType, CharacterType, ClassType,
           FieldType, HeapType, fir::IntegerType, LenType, LogicalType,
           LLVMPointerType, PointerType, RecordType, ReferenceType,
           SequenceType, ShapeType, ShapeShiftType, ShiftType, SliceType,
           TypeDescType, fir::VectorType, fir::DummyScopeType>();
  fir::ReferenceType::attachInterface<
      OpenMPPointerLikeModel<fir::ReferenceType>>(*getContext());
  fir::PointerType::attachInterface<OpenMPPointerLikeModel<fir::PointerType>>(
      *getContext());
  fir::HeapType::attachInterface<OpenMPPointerLikeModel<fir::HeapType>>(
      *getContext());
  fir::LLVMPointerType::attachInterface<
      OpenMPPointerLikeModel<fir::LLVMPointerType>>(*getContext());
}

std::optional<std::pair<uint64_t, unsigned short>>
fir::getTypeSizeAndAlignment(aiir::Location loc, aiir::Type ty,
                             const aiir::DataLayout &dl,
                             const fir::KindMapping &kindMap) {
  if (ty.isIntOrIndexOrFloat() ||
      aiir::isa<aiir::ComplexType, aiir::VectorType,
                aiir::DataLayoutTypeInterface>(ty)) {
    llvm::TypeSize size = dl.getTypeSize(ty);
    unsigned short alignment = dl.getTypeABIAlignment(ty);
    return std::pair{size, alignment};
  }
  if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(ty)) {
    auto result = getTypeSizeAndAlignment(loc, seqTy.getEleTy(), dl, kindMap);
    if (!result)
      return result;
    auto [eleSize, eleAlign] = *result;
    std::uint64_t size =
        llvm::alignTo(eleSize, eleAlign) * seqTy.getConstantArraySize();
    return std::pair{size, eleAlign};
  }
  if (auto recTy = aiir::dyn_cast<fir::RecordType>(ty)) {
    std::uint64_t size = 0;
    unsigned short align = 1;
    for (auto component : recTy.getTypeList()) {
      auto result = getTypeSizeAndAlignment(loc, component.second, dl, kindMap);
      if (!result)
        return result;
      auto [compSize, compAlign] = *result;
      size =
          llvm::alignTo(size, compAlign) + llvm::alignTo(compSize, compAlign);
      align = std::max(align, compAlign);
    }
    return std::pair{size, align};
  }
  if (auto logical = aiir::dyn_cast<fir::LogicalType>(ty)) {
    aiir::Type intTy = aiir::IntegerType::get(
        logical.getContext(), kindMap.getLogicalBitsize(logical.getFKind()));
    return getTypeSizeAndAlignment(loc, intTy, dl, kindMap);
  }
  if (auto character = aiir::dyn_cast<fir::CharacterType>(ty)) {
    aiir::Type intTy = aiir::IntegerType::get(
        character.getContext(),
        kindMap.getCharacterBitsize(character.getFKind()));
    auto result = getTypeSizeAndAlignment(loc, intTy, dl, kindMap);
    if (!result)
      return result;
    auto [compSize, compAlign] = *result;
    if (character.hasConstantLen())
      compSize *= character.getLen();
    return std::pair{compSize, compAlign};
  }
  return std::nullopt;
}

std::pair<std::uint64_t, unsigned short>
fir::getTypeSizeAndAlignmentOrCrash(aiir::Location loc, aiir::Type ty,
                                    const aiir::DataLayout &dl,
                                    const fir::KindMapping &kindMap) {
  std::optional<std::pair<uint64_t, unsigned short>> result =
      getTypeSizeAndAlignment(loc, ty, dl, kindMap);
  if (result)
    return *result;
  TODO(loc, "computing size of a component");
}
