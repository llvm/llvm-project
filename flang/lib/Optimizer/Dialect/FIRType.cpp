//===-- FIRType.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace fir;

namespace {

template <typename TYPE>
TYPE parseIntSingleton(mlir::DialectAsmParser &parser) {
  int kind = 0;
  if (parser.parseLess() || parser.parseInteger(kind) ||
      parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation(), "kind value expected");
    return {};
  }
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
TYPE parseTypeSingleton(mlir::DialectAsmParser &parser, mlir::Location) {
  mlir::Type ty;
  if (parser.parseLess() || parser.parseType(ty) || parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation(), "type expected");
    return {};
  }
  return TYPE::get(ty);
}

// `box` `<` type (',' affine-map)? `>`
BoxType parseBox(mlir::DialectAsmParser &parser, mlir::Location loc) {
  mlir::Type ofTy;
  if (parser.parseLess() || parser.parseType(ofTy)) {
    parser.emitError(parser.getCurrentLocation(), "expected type parameter");
    return {};
  }

  mlir::AffineMapAttr map;
  if (!parser.parseOptionalComma())
    if (parser.parseAttribute(map)) {
      parser.emitError(parser.getCurrentLocation(), "expected affine map");
      return {};
    }
  if (parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation(), "expected '>'");
    return {};
  }
  return BoxType::get(ofTy, map);
}

// `boxchar` `<` kind `>`
BoxCharType parseBoxChar(mlir::DialectAsmParser &parser) {
  return parseKindSingleton<BoxCharType>(parser);
}

// `boxproc` `<` return-type `>`
BoxProcType parseBoxProc(mlir::DialectAsmParser &parser, mlir::Location loc) {
  return parseTypeSingleton<BoxProcType>(parser, loc);
}

// `char` `<` kind `>`
CharacterType parseCharacter(mlir::DialectAsmParser &parser) {
  return parseKindSingleton<CharacterType>(parser);
}

// `complex` `<` kind `>`
CplxType parseComplex(mlir::DialectAsmParser &parser) {
  return parseKindSingleton<CplxType>(parser);
}

// `dims` `<` rank `>`
DimsType parseDims(mlir::DialectAsmParser &parser) {
  return parseRankSingleton<DimsType>(parser);
}

// `field`
FieldType parseField(mlir::DialectAsmParser &parser) {
  return FieldType::get(parser.getBuilder().getContext());
}

// `heap` `<` type `>`
HeapType parseHeap(mlir::DialectAsmParser &parser, mlir::Location loc) {
  return parseTypeSingleton<HeapType>(parser, loc);
}

// `int` `<` kind `>`
IntType parseInteger(mlir::DialectAsmParser &parser) {
  return parseKindSingleton<IntType>(parser);
}

// `len`
LenType parseLen(mlir::DialectAsmParser &parser) {
  return LenType::get(parser.getBuilder().getContext());
}

// `logical` `<` kind `>`
LogicalType parseLogical(mlir::DialectAsmParser &parser) {
  return parseKindSingleton<LogicalType>(parser);
}

// `ptr` `<` type `>`
PointerType parsePointer(mlir::DialectAsmParser &parser, mlir::Location loc) {
  return parseTypeSingleton<PointerType>(parser, loc);
}

// `real` `<` kind `>`
RealType parseReal(mlir::DialectAsmParser &parser) {
  return parseKindSingleton<RealType>(parser);
}

// `ref` `<` type `>`
ReferenceType parseReference(mlir::DialectAsmParser &parser,
                             mlir::Location loc) {
  return parseTypeSingleton<ReferenceType>(parser, loc);
}

// `tdesc` `<` type `>`
TypeDescType parseTypeDesc(mlir::DialectAsmParser &parser, mlir::Location loc) {
  return parseTypeSingleton<TypeDescType>(parser, loc);
}

// `void`
mlir::Type parseVoid(mlir::DialectAsmParser &parser) {
  return parser.getBuilder().getNoneType();
}

// `array` `<` `*` | bounds (`x` bounds)* `:` type (',' affine-map)? `>`
// bounds ::= `?` | int-lit
SequenceType parseSequence(mlir::DialectAsmParser &parser, mlir::Location) {
  if (parser.parseLess()) {
    parser.emitError(parser.getNameLoc(), "expecting '<'");
    return {};
  }
  SequenceType::Shape shape;
  if (parser.parseOptionalStar()) {
    if (parser.parseDimensionList(shape, true)) {
      parser.emitError(parser.getNameLoc(), "invalid shape");
      return {};
    }
  } else if (parser.parseColon()) {
    parser.emitError(parser.getNameLoc(), "expected ':'");
    return {};
  }
  mlir::Type eleTy;
  if (parser.parseType(eleTy) || parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expecting element type");
    return {};
  }
  mlir::AffineMapAttr map;
  if (!parser.parseOptionalComma())
    if (parser.parseAttribute(map)) {
      parser.emitError(parser.getNameLoc(), "expecting affine map");
      return {};
    }
  return SequenceType::get(shape, eleTy, map);
}

/// Is `ty` a standard or FIR integer type?
static bool isaIntegerType(mlir::Type ty) {
  // TODO: why aren't we using isa_integer? investigatation required.
  return ty.isa<mlir::IntegerType>() || ty.isa<fir::IntType>();
}

bool verifyRecordMemberType(mlir::Type ty) {
  return !(ty.isa<BoxType>() || ty.isa<BoxCharType>() ||
           ty.isa<BoxProcType>() || ty.isa<DimsType>() || ty.isa<FieldType>() ||
           ty.isa<LenType>() || ty.isa<ReferenceType>() ||
           ty.isa<TypeDescType>());
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

// Fortran derived type
// `type` `<` name
//           (`(` id `:` type (`,` id `:` type)* `)`)?
//           (`{` id `:` type (`,` id `:` type)* `}`)? '>'
RecordType parseDerived(mlir::DialectAsmParser &parser, mlir::Location) {
  llvm::StringRef name;
  if (parser.parseLess() || parser.parseKeyword(&name)) {
    parser.emitError(parser.getNameLoc(),
                     "expected a identifier as name of derived type");
    return {};
  }
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
    if (parser.parseRParen()) {
      parser.emitError(parser.getNameLoc(), "expected ')'");
      return {};
    }
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
    if (parser.parseRBrace()) {
      parser.emitError(parser.getNameLoc(), "expected '}'");
      return {};
    }
  }

  if (parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expected '>' in type type");
    return {};
  }

  if (lenParamList.empty() && typeList.empty())
    return result;

  result.finalize(lenParamList, typeList);
  return verifyDerived(parser, result, lenParamList, typeList);
}

#ifndef NDEBUG
// !fir.ptr<X> and !fir.heap<X> where X is !fir.ptr, !fir.heap, or !fir.ref
// is undefined and disallowed.
inline bool singleIndirectionLevel(mlir::Type ty) {
  return !fir::isa_ref_type(ty);
}
#endif

} // namespace

// Implementation of the thin interface from dialect to type parser

mlir::Type fir::parseFirType(FIROpsDialect *, mlir::DialectAsmParser &parser) {
  llvm::StringRef typeNameLit;
  if (mlir::failed(parser.parseKeyword(&typeNameLit)))
    return {};

  auto loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  if (typeNameLit == "array")
    return parseSequence(parser, loc);
  if (typeNameLit == "box")
    return parseBox(parser, loc);
  if (typeNameLit == "boxchar")
    return parseBoxChar(parser);
  if (typeNameLit == "boxproc")
    return parseBoxProc(parser, loc);
  if (typeNameLit == "char")
    return parseCharacter(parser);
  if (typeNameLit == "complex")
    return parseComplex(parser);
  if (typeNameLit == "dims")
    return parseDims(parser);
  if (typeNameLit == "field")
    return parseField(parser);
  if (typeNameLit == "heap")
    return parseHeap(parser, loc);
  if (typeNameLit == "int")
    return parseInteger(parser);
  if (typeNameLit == "len")
    return parseLen(parser);
  if (typeNameLit == "logical")
    return parseLogical(parser);
  if (typeNameLit == "ptr")
    return parsePointer(parser, loc);
  if (typeNameLit == "real")
    return parseReal(parser);
  if (typeNameLit == "ref")
    return parseReference(parser, loc);
  if (typeNameLit == "tdesc")
    return parseTypeDesc(parser, loc);
  if (typeNameLit == "type")
    return parseDerived(parser, loc);
  if (typeNameLit == "void")
    return parseVoid(parser);

  parser.emitError(parser.getNameLoc(), "unknown FIR type " + typeNameLit);
  return {};
}

namespace fir {
namespace detail {

// Type storage classes

/// `CHARACTER` storage
struct CharacterTypeStorage : public mlir::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static CharacterTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                         KindTy kind) {
    auto *storage = allocator.allocate<CharacterTypeStorage>();
    return new (storage) CharacterTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  CharacterTypeStorage() = delete;
  explicit CharacterTypeStorage(KindTy kind) : kind{kind} {}
};

struct DimsTypeStorage : public mlir::TypeStorage {
  using KeyTy = unsigned;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getRank(); }

  static DimsTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    unsigned rank) {
    auto *storage = allocator.allocate<DimsTypeStorage>();
    return new (storage) DimsTypeStorage{rank};
  }

  unsigned getRank() const { return rank; }

protected:
  unsigned rank;

private:
  DimsTypeStorage() = delete;
  explicit DimsTypeStorage(unsigned rank) : rank{rank} {}
};

/// The type of a derived type part reference
struct FieldTypeStorage : public mlir::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &) { return llvm::hash_combine(0); }

  bool operator==(const KeyTy &) const { return true; }

  static FieldTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     KindTy) {
    auto *storage = allocator.allocate<FieldTypeStorage>();
    return new (storage) FieldTypeStorage{0};
  }

private:
  FieldTypeStorage() = delete;
  explicit FieldTypeStorage(KindTy) {}
};

/// The type of a derived type LEN parameter reference
struct LenTypeStorage : public mlir::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &) { return llvm::hash_combine(0); }

  bool operator==(const KeyTy &) const { return true; }

  static LenTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   KindTy) {
    auto *storage = allocator.allocate<LenTypeStorage>();
    return new (storage) LenTypeStorage{0};
  }

private:
  LenTypeStorage() = delete;
  explicit LenTypeStorage(KindTy) {}
};

/// `LOGICAL` storage
struct LogicalTypeStorage : public mlir::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static LogicalTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       KindTy kind) {
    auto *storage = allocator.allocate<LogicalTypeStorage>();
    return new (storage) LogicalTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  LogicalTypeStorage() = delete;
  explicit LogicalTypeStorage(KindTy kind) : kind{kind} {}
};

/// `INTEGER` storage
struct IntTypeStorage : public mlir::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static IntTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   KindTy kind) {
    auto *storage = allocator.allocate<IntTypeStorage>();
    return new (storage) IntTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  IntTypeStorage() = delete;
  explicit IntTypeStorage(KindTy kind) : kind{kind} {}
};

/// `COMPLEX` storage
struct CplxTypeStorage : public mlir::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static CplxTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    KindTy kind) {
    auto *storage = allocator.allocate<CplxTypeStorage>();
    return new (storage) CplxTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  CplxTypeStorage() = delete;
  explicit CplxTypeStorage(KindTy kind) : kind{kind} {}
};

/// `REAL` storage (for reals of unsupported sizes)
struct RealTypeStorage : public mlir::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static RealTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    KindTy kind) {
    auto *storage = allocator.allocate<RealTypeStorage>();
    return new (storage) RealTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  RealTypeStorage() = delete;
  explicit RealTypeStorage(KindTy kind) : kind{kind} {}
};

/// Boxed object (a Fortran descriptor)
struct BoxTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<mlir::Type, mlir::AffineMapAttr>;

  static unsigned hashKey(const KeyTy &key) {
    auto hashVal{llvm::hash_combine(std::get<mlir::Type>(key))};
    return llvm::hash_combine(
        hashVal, llvm::hash_combine(std::get<mlir::AffineMapAttr>(key)));
  }

  bool operator==(const KeyTy &key) const {
    return std::get<mlir::Type>(key) == getElementType() &&
           std::get<mlir::AffineMapAttr>(key) == getLayoutMap();
  }

  static BoxTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    auto *storage = allocator.allocate<BoxTypeStorage>();
    return new (storage) BoxTypeStorage{std::get<mlir::Type>(key),
                                        std::get<mlir::AffineMapAttr>(key)};
  }

  mlir::Type getElementType() const { return eleTy; }
  mlir::AffineMapAttr getLayoutMap() const { return map; }

protected:
  mlir::Type eleTy;
  mlir::AffineMapAttr map;

private:
  BoxTypeStorage() = delete;
  explicit BoxTypeStorage(mlir::Type eleTy, mlir::AffineMapAttr map)
      : eleTy{eleTy}, map{map} {}
};

/// Boxed CHARACTER object type
struct BoxCharTypeStorage : public mlir::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static BoxCharTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       KindTy kind) {
    auto *storage = allocator.allocate<BoxCharTypeStorage>();
    return new (storage) BoxCharTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

  // a !fir.boxchar<k> always wraps a !fir.char<k>
  CharacterType getElementType(mlir::MLIRContext *ctxt) const {
    return CharacterType::get(ctxt, getFKind());
  }

protected:
  KindTy kind;

private:
  BoxCharTypeStorage() = delete;
  explicit BoxCharTypeStorage(KindTy kind) : kind{kind} {}
};

/// Boxed PROCEDURE POINTER object type
struct BoxProcTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static BoxProcTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       mlir::Type eleTy) {
    assert(eleTy && "element type is null");
    auto *storage = allocator.allocate<BoxProcTypeStorage>();
    return new (storage) BoxProcTypeStorage{eleTy};
  }

  mlir::Type getElementType() const { return eleTy; }

protected:
  mlir::Type eleTy;

private:
  BoxProcTypeStorage() = delete;
  explicit BoxProcTypeStorage(mlir::Type eleTy) : eleTy{eleTy} {}
};

/// Pointer-like object storage
struct ReferenceTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static ReferenceTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                         mlir::Type eleTy) {
    assert(eleTy && "element type is null");
    auto *storage = allocator.allocate<ReferenceTypeStorage>();
    return new (storage) ReferenceTypeStorage{eleTy};
  }

  mlir::Type getElementType() const { return eleTy; }

protected:
  mlir::Type eleTy;

private:
  ReferenceTypeStorage() = delete;
  explicit ReferenceTypeStorage(mlir::Type eleTy) : eleTy{eleTy} {}
};

/// Pointer object storage
struct PointerTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static PointerTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       mlir::Type eleTy) {
    assert(eleTy && "element type is null");
    auto *storage = allocator.allocate<PointerTypeStorage>();
    return new (storage) PointerTypeStorage{eleTy};
  }

  mlir::Type getElementType() const { return eleTy; }

protected:
  mlir::Type eleTy;

private:
  PointerTypeStorage() = delete;
  explicit PointerTypeStorage(mlir::Type eleTy) : eleTy{eleTy} {}
};

/// Heap memory reference object storage
struct HeapTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static HeapTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                    mlir::Type eleTy) {
    assert(eleTy && "element type is null");
    auto *storage = allocator.allocate<HeapTypeStorage>();
    return new (storage) HeapTypeStorage{eleTy};
  }

  mlir::Type getElementType() const { return eleTy; }

protected:
  mlir::Type eleTy;

private:
  HeapTypeStorage() = delete;
  explicit HeapTypeStorage(mlir::Type eleTy) : eleTy{eleTy} {}
};

/// Sequence-like object storage
struct SequenceTypeStorage : public mlir::TypeStorage {
  using KeyTy =
      std::tuple<SequenceType::Shape, mlir::Type, mlir::AffineMapAttr>;

  static unsigned hashKey(const KeyTy &key) {
    auto shapeHash{hash_value(std::get<SequenceType::Shape>(key))};
    shapeHash = llvm::hash_combine(shapeHash, std::get<mlir::Type>(key));
    return llvm::hash_combine(shapeHash, std::get<mlir::AffineMapAttr>(key));
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy{getShape(), getElementType(), getLayoutMap()};
  }

  static SequenceTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    auto *storage = allocator.allocate<SequenceTypeStorage>();
    return new (storage) SequenceTypeStorage{
        std::get<SequenceType::Shape>(key), std::get<mlir::Type>(key),
        std::get<mlir::AffineMapAttr>(key)};
  }

  SequenceType::Shape getShape() const { return shape; }
  mlir::Type getElementType() const { return eleTy; }
  mlir::AffineMapAttr getLayoutMap() const { return map; }

protected:
  SequenceType::Shape shape;
  mlir::Type eleTy;
  mlir::AffineMapAttr map;

private:
  SequenceTypeStorage() = delete;
  explicit SequenceTypeStorage(const SequenceType::Shape &shape,
                               mlir::Type eleTy, mlir::AffineMapAttr map)
      : shape{shape}, eleTy{eleTy}, map{map} {}
};

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

/// Type descriptor type storage
struct TypeDescTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getOfType(); }

  static TypeDescTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                        mlir::Type ofTy) {
    assert(ofTy && "descriptor type is null");
    auto *storage = allocator.allocate<TypeDescTypeStorage>();
    return new (storage) TypeDescTypeStorage{ofTy};
  }

  // The type described by this type descriptor instance
  mlir::Type getOfType() const { return ofTy; }

protected:
  mlir::Type ofTy;

private:
  TypeDescTypeStorage() = delete;
  explicit TypeDescTypeStorage(mlir::Type ofTy) : ofTy{ofTy} {}
};

} // namespace detail

template <typename A, typename B>
bool inbounds(A v, B lb, B ub) {
  return v >= lb && v < ub;
}

bool isa_fir_type(mlir::Type t) {
  return inbounds(t.getKind(), mlir::Type::FIRST_FIR_TYPE,
                  mlir::Type::LAST_FIR_TYPE);
}

bool isa_std_type(mlir::Type t) {
  return inbounds(t.getKind(), mlir::Type::FIRST_STANDARD_TYPE,
                  mlir::Type::LAST_STANDARD_TYPE);
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
  return t.isa<ReferenceType>() || isa_box_type(t);
}

bool isa_aggregate(mlir::Type t) {
  return t.isa<SequenceType>() || t.isa<RecordType>();
}

mlir::Type dyn_cast_ptrEleTy(mlir::Type t) {
  return llvm::TypeSwitch<mlir::Type, mlir::Type>(t)
      .Case<fir::ReferenceType, fir::PointerType, fir::HeapType>(
          [](auto p) { return p.getEleTy(); })
      .Default([](mlir::Type) { return mlir::Type{}; });
}

} // namespace fir

// CHARACTER

CharacterType fir::CharacterType::get(mlir::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_CHARACTER, kind);
}

int fir::CharacterType::getFKind() const { return getImpl()->getFKind(); }

// Dims

DimsType fir::DimsType::get(mlir::MLIRContext *ctxt, unsigned rank) {
  return Base::get(ctxt, FIR_DIMS, rank);
}

unsigned fir::DimsType::getRank() const { return getImpl()->getRank(); }

// Field

FieldType fir::FieldType::get(mlir::MLIRContext *ctxt) {
  return Base::get(ctxt, FIR_FIELD, 0);
}

// Len

LenType fir::LenType::get(mlir::MLIRContext *ctxt) {
  return Base::get(ctxt, FIR_LEN, 0);
}

// LOGICAL

LogicalType fir::LogicalType::get(mlir::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_LOGICAL, kind);
}

int fir::LogicalType::getFKind() const { return getImpl()->getFKind(); }

// INTEGER

IntType fir::IntType::get(mlir::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_INT, kind);
}

int fir::IntType::getFKind() const { return getImpl()->getFKind(); }

// COMPLEX

CplxType fir::CplxType::get(mlir::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_COMPLEX, kind);
}

mlir::Type fir::CplxType::getElementType() const {
  return fir::RealType::get(getContext(), getFKind());
}

KindTy fir::CplxType::getFKind() const { return getImpl()->getFKind(); }

// REAL

RealType fir::RealType::get(mlir::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_REAL, kind);
}

int fir::RealType::getFKind() const { return getImpl()->getFKind(); }

// Box<T>

BoxType fir::BoxType::get(mlir::Type elementType, mlir::AffineMapAttr map) {
  return Base::get(elementType.getContext(), FIR_BOX, elementType, map);
}

mlir::Type fir::BoxType::getEleTy() const {
  return getImpl()->getElementType();
}

mlir::AffineMapAttr fir::BoxType::getLayoutMap() const {
  return getImpl()->getLayoutMap();
}

mlir::LogicalResult
fir::BoxType::verifyConstructionInvariants(mlir::Location, mlir::Type eleTy,
                                           mlir::AffineMapAttr map) {
  // TODO
  return mlir::success();
}

// BoxChar<C>

BoxCharType fir::BoxCharType::get(mlir::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_BOXCHAR, kind);
}

CharacterType fir::BoxCharType::getEleTy() const {
  return getImpl()->getElementType(getContext());
}

// BoxProc<T>

BoxProcType fir::BoxProcType::get(mlir::Type elementType) {
  return Base::get(elementType.getContext(), FIR_BOXPROC, elementType);
}

mlir::Type fir::BoxProcType::getEleTy() const {
  return getImpl()->getElementType();
}

mlir::LogicalResult
fir::BoxProcType::verifyConstructionInvariants(mlir::Location loc,
                                               mlir::Type eleTy) {
  if (eleTy.isa<mlir::FunctionType>())
    return mlir::success();
  if (auto refTy = eleTy.dyn_cast<ReferenceType>())
    if (refTy.isa<mlir::FunctionType>())
      return mlir::success();
  return mlir::emitError(loc, "invalid type for boxproc") << eleTy << '\n';
}

// Reference<T>

ReferenceType fir::ReferenceType::get(mlir::Type elementType) {
  return Base::get(elementType.getContext(), FIR_REFERENCE, elementType);
}

mlir::Type fir::ReferenceType::getEleTy() const {
  return getImpl()->getElementType();
}

mlir::LogicalResult
fir::ReferenceType::verifyConstructionInvariants(mlir::Location loc,
                                                 mlir::Type eleTy) {
  if (eleTy.isa<DimsType>() || eleTy.isa<FieldType>() || eleTy.isa<LenType>() ||
      eleTy.isa<ReferenceType>() || eleTy.isa<TypeDescType>())
    return mlir::emitError(loc, "cannot build a reference to type: ")
           << eleTy << '\n';
  return mlir::success();
}

// Pointer<T>

PointerType fir::PointerType::get(mlir::Type elementType) {
  assert(singleIndirectionLevel(elementType) && "invalid element type");
  return Base::get(elementType.getContext(), FIR_POINTER, elementType);
}

mlir::Type fir::PointerType::getEleTy() const {
  return getImpl()->getElementType();
}

static bool canBePointerOrHeapElementType(mlir::Type eleTy) {
  return eleTy.isa<BoxType>() || eleTy.isa<BoxCharType>() ||
         eleTy.isa<BoxProcType>() || eleTy.isa<DimsType>() ||
         eleTy.isa<FieldType>() || eleTy.isa<LenType>() ||
         eleTy.isa<HeapType>() || eleTy.isa<PointerType>() ||
         eleTy.isa<ReferenceType>() || eleTy.isa<TypeDescType>();
}

mlir::LogicalResult
fir::PointerType::verifyConstructionInvariants(mlir::Location loc,
                                               mlir::Type eleTy) {
  if (canBePointerOrHeapElementType(eleTy))
    return mlir::emitError(loc, "cannot build a pointer to type: ")
           << eleTy << '\n';
  return mlir::success();
}

// Heap<T>

HeapType fir::HeapType::get(mlir::Type elementType) {
  assert(singleIndirectionLevel(elementType) && "invalid element type");
  return Base::get(elementType.getContext(), FIR_HEAP, elementType);
}

mlir::Type fir::HeapType::getEleTy() const {
  return getImpl()->getElementType();
}

mlir::LogicalResult
fir::HeapType::verifyConstructionInvariants(mlir::Location loc,
                                            mlir::Type eleTy) {
  if (canBePointerOrHeapElementType(eleTy))
    return mlir::emitError(loc, "cannot build a heap pointer to type: ")
           << eleTy << '\n';
  return mlir::success();
}

// Sequence<T>

SequenceType fir::SequenceType::get(const Shape &shape, mlir::Type elementType,
                                    mlir::AffineMapAttr map) {
  auto *ctxt = elementType.getContext();
  return Base::get(ctxt, FIR_SEQUENCE, shape, elementType, map);
}

mlir::Type fir::SequenceType::getEleTy() const {
  return getImpl()->getElementType();
}

mlir::AffineMapAttr fir::SequenceType::getLayoutMap() const {
  return getImpl()->getLayoutMap();
}

SequenceType::Shape fir::SequenceType::getShape() const {
  return getImpl()->getShape();
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
  for (unsigned i{rows}, size{dim}; i < size; ++i)
    if (shape[i] != getUnknownExtent())
      return false;
  return true;
}

mlir::LogicalResult fir::SequenceType::verifyConstructionInvariants(
    mlir::Location loc, const SequenceType::Shape &shape, mlir::Type eleTy,
    mlir::AffineMapAttr map) {
  // DIMENSION attribute can only be applied to an intrinsic or record type
  if (eleTy.isa<BoxType>() || eleTy.isa<BoxCharType>() ||
      eleTy.isa<BoxProcType>() || eleTy.isa<DimsType>() ||
      eleTy.isa<FieldType>() || eleTy.isa<LenType>() || eleTy.isa<HeapType>() ||
      eleTy.isa<PointerType>() || eleTy.isa<ReferenceType>() ||
      eleTy.isa<TypeDescType>() || eleTy.isa<SequenceType>())
    return mlir::emitError(loc, "cannot build an array of this element type: ")
           << eleTy << '\n';
  return mlir::success();
}

// compare if two shapes are equivalent
bool fir::operator==(const SequenceType::Shape &sh_1,
                     const SequenceType::Shape &sh_2) {
  if (sh_1.size() != sh_2.size())
    return false;
  auto e = sh_1.size();
  for (decltype(e) i = 0; i != e; ++i)
    if (sh_1[i] != sh_2[i])
      return false;
  return true;
}

// compute the hash of a Shape
llvm::hash_code fir::hash_value(const SequenceType::Shape &sh) {
  if (sh.size()) {
    return llvm::hash_combine_range(sh.begin(), sh.end());
  }
  return llvm::hash_combine(0);
}

/// RecordType
///
/// This type captures a Fortran "derived type"

RecordType fir::RecordType::get(mlir::MLIRContext *ctxt, llvm::StringRef name) {
  return Base::get(ctxt, FIR_DERIVED, name);
}

void fir::RecordType::finalize(llvm::ArrayRef<TypePair> lenPList,
                               llvm::ArrayRef<TypePair> typeList) {
  getImpl()->finalize(lenPList, typeList);
}

llvm::StringRef fir::RecordType::getName() { return getImpl()->getName(); }

RecordType::TypeList fir::RecordType::getTypeList() {
  return getImpl()->getTypeList();
}

RecordType::TypeList fir::RecordType::getLenParamList() {
  return getImpl()->getLenParamList();
}

detail::RecordTypeStorage const *fir::RecordType::uniqueKey() const {
  return getImpl();
}

mlir::LogicalResult
fir::RecordType::verifyConstructionInvariants(mlir::Location loc,
                                              llvm::StringRef name) {
  if (name.size() == 0)
    return mlir::emitError(loc, "record types must have a name");
  return mlir::success();
}

mlir::Type fir::RecordType::getType(llvm::StringRef ident) {
  for (auto f : getTypeList())
    if (ident == f.first)
      return f.second;
  return {};
}

/// Type descriptor type
///
/// This is the type of a type descriptor object (similar to a class instance)

TypeDescType fir::TypeDescType::get(mlir::Type ofType) {
  assert(!ofType.isa<ReferenceType>());
  return Base::get(ofType.getContext(), FIR_TYPEDESC, ofType);
}

mlir::Type fir::TypeDescType::getOfTy() const { return getImpl()->getOfType(); }

mlir::LogicalResult
fir::TypeDescType::verifyConstructionInvariants(mlir::Location loc,
                                                mlir::Type eleTy) {
  if (eleTy.isa<BoxType>() || eleTy.isa<BoxCharType>() ||
      eleTy.isa<BoxProcType>() || eleTy.isa<DimsType>() ||
      eleTy.isa<FieldType>() || eleTy.isa<LenType>() ||
      eleTy.isa<ReferenceType>() || eleTy.isa<TypeDescType>())
    return mlir::emitError(loc, "cannot build a type descriptor of type: ")
           << eleTy << '\n';
  return mlir::success();
}

namespace {

void printBounds(llvm::raw_ostream &os, const SequenceType::Shape &bounds) {
  os << '<';
  for (auto &b : bounds) {
    if (b >= 0) {
      os << b << 'x';
    } else {
      os << "?x";
    }
  }
}

llvm::SmallPtrSet<detail::RecordTypeStorage const *, 4> recordTypeVisited;

} // namespace

void fir::verifyIntegralType(mlir::Type type) {
  if (isaIntegerType(type) || type.isa<mlir::IndexType>())
    return;
  llvm::report_fatal_error("expected integral type");
}

void fir::printFirType(FIROpsDialect *, mlir::Type ty,
                       mlir::DialectAsmPrinter &p) {
  auto &os = p.getStream();
  switch (ty.getKind()) {
  case fir::FIR_BOX: {
    auto type = ty.cast<BoxType>();
    os << "box<";
    p.printType(type.getEleTy());
    if (auto map = type.getLayoutMap()) {
      os << ", ";
      p.printAttribute(map);
    }
    os << '>';
  } break;
  case fir::FIR_BOXCHAR: {
    auto type = ty.cast<BoxCharType>().getEleTy();
    os << "boxchar<" << type.cast<fir::CharacterType>().getFKind() << '>';
  } break;
  case fir::FIR_BOXPROC:
    os << "boxproc<";
    p.printType(ty.cast<BoxProcType>().getEleTy());
    os << '>';
    break;
  case fir::FIR_CHARACTER: // intrinsic
    os << "char<" << ty.cast<CharacterType>().getFKind() << '>';
    break;
  case fir::FIR_COMPLEX: // intrinsic
    os << "complex<" << ty.cast<CplxType>().getFKind() << '>';
    break;
  case fir::FIR_DERIVED: { // derived
    auto type = ty.cast<fir::RecordType>();
    os << "type<" << type.getName();
    if (!recordTypeVisited.count(type.uniqueKey())) {
      recordTypeVisited.insert(type.uniqueKey());
      if (type.getLenParamList().size()) {
        char ch = '(';
        for (auto p : type.getLenParamList()) {
          os << ch << p.first << ':';
          p.second.print(os);
          ch = ',';
        }
        os << ')';
      }
      if (type.getTypeList().size()) {
        char ch = '{';
        for (auto p : type.getTypeList()) {
          os << ch << p.first << ':';
          p.second.print(os);
          ch = ',';
        }
        os << '}';
      }
      recordTypeVisited.erase(type.uniqueKey());
    }
    os << '>';
  } break;
  case fir::FIR_DIMS:
    os << "dims<" << ty.cast<DimsType>().getRank() << '>';
    break;
  case fir::FIR_FIELD:
    os << "field";
    break;
  case fir::FIR_HEAP:
    os << "heap<";
    p.printType(ty.cast<HeapType>().getEleTy());
    os << '>';
    break;
  case fir::FIR_INT: // intrinsic
    os << "int<" << ty.cast<fir::IntType>().getFKind() << '>';
    break;
  case fir::FIR_LEN:
    os << "len";
    break;
  case fir::FIR_LOGICAL: // intrinsic
    os << "logical<" << ty.cast<LogicalType>().getFKind() << '>';
    break;
  case fir::FIR_POINTER:
    os << "ptr<";
    p.printType(ty.cast<PointerType>().getEleTy());
    os << '>';
    break;
  case fir::FIR_REAL: // intrinsic
    os << "real<" << ty.cast<fir::RealType>().getFKind() << '>';
    break;
  case fir::FIR_REFERENCE:
    os << "ref<";
    p.printType(ty.cast<ReferenceType>().getEleTy());
    os << '>';
    break;
  case fir::FIR_SEQUENCE: {
    os << "array";
    auto type = ty.cast<SequenceType>();
    auto shape = type.getShape();
    if (shape.size()) {
      printBounds(os, shape);
    } else {
      os << "<*:";
    }
    p.printType(ty.cast<SequenceType>().getEleTy());
    if (auto map = type.getLayoutMap()) {
      os << ", ";
      map.print(os);
    }
    os << '>';
  } break;
  case fir::FIR_TYPEDESC:
    os << "tdesc<";
    p.printType(ty.cast<TypeDescType>().getOfTy());
    os << '>';
    break;
  }
}
