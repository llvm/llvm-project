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

// `boxchar` `<` kind `>`
BoxCharType parseBoxChar(mlir::DialectAsmParser &parser) {
  return parseKindSingleton<BoxCharType>(parser);
}

// `boxproc` `<` return-type `>`
BoxProcType parseBoxProc(mlir::DialectAsmParser &parser, mlir::Location loc) {
  return parseTypeSingleton<BoxProcType>(parser, loc);
}

// `char` `<` kind [`,` `len`] `>`
CharacterType parseCharacter(mlir::DialectAsmParser &parser) {
  int kind = 0;
  if (parser.parseLess() || parser.parseInteger(kind)) {
    parser.emitError(parser.getCurrentLocation(), "kind value expected");
    return {};
  }
  CharacterType::LenType len = 1;
  if (mlir::succeeded(parser.parseOptionalComma())) {
    if (mlir::succeeded(parser.parseOptionalQuestion())) {
      len = fir::CharacterType::unknownLen();
    } else if (!mlir::succeeded(parser.parseInteger(len))) {
      parser.emitError(parser.getCurrentLocation(), "len value expected");
      return {};
    }
  }
  if (parser.parseGreater())
    return {};
  return CharacterType::get(parser.getBuilder().getContext(), kind, len);
}

// `complex` `<` kind `>`
fir::ComplexType parseComplex(mlir::DialectAsmParser &parser) {
  return parseKindSingleton<fir::ComplexType>(parser);
}

// `slice` `<` rank `>`
SliceType parseSlice(mlir::DialectAsmParser &parser) {
  return parseRankSingleton<SliceType>(parser);
}

// `heap` `<` type `>`
HeapType parseHeap(mlir::DialectAsmParser &parser, mlir::Location loc) {
  return parseTypeSingleton<HeapType>(parser, loc);
}

// `int` `<` kind `>`
fir::IntegerType parseInteger(mlir::DialectAsmParser &parser) {
  return parseKindSingleton<fir::IntegerType>(parser);
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

// `vector` `<` len `:` type `>`
fir::VectorType parseVector(mlir::DialectAsmParser &parser,
                            mlir::Location loc) {
  int64_t len = 0;
  mlir::Type eleTy;
  if (parser.parseLess() || parser.parseInteger(len) || parser.parseColon() ||
      parser.parseType(eleTy) || parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "invalid vector type");
    return {};
  }
  return fir::VectorType::get(len, eleTy);
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
    if (parser.parseDimensionList(shape, /*allowDynamic=*/true)) {
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
  return ty.isa<mlir::IntegerType>() || ty.isa<fir::IntegerType>();
}

bool verifyRecordMemberType(mlir::Type ty) {
  return !(ty.isa<BoxType>() || ty.isa<BoxCharType>() ||
           ty.isa<BoxProcType>() || ty.isa<ShapeType>() ||
           ty.isa<ShapeShiftType>() || ty.isa<SliceType>() ||
           ty.isa<FieldType>() || ty.isa<LenType>() ||
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

mlir::Type fir::parseFirType(FIROpsDialect *dialect,
                             mlir::DialectAsmParser &parser) {
  llvm::StringRef typeNameLit;
  if (mlir::failed(parser.parseKeyword(&typeNameLit)))
    return {};

  auto loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  if (typeNameLit == "array")
    return parseSequence(parser, loc);
  if (typeNameLit == "box")
    return generatedTypeParser(dialect->getContext(), parser, typeNameLit);
  if (typeNameLit == "boxchar")
    return parseBoxChar(parser);
  if (typeNameLit == "boxproc")
    return parseBoxProc(parser, loc);
  if (typeNameLit == "char")
    return parseCharacter(parser);
  if (typeNameLit == "complex")
    return parseComplex(parser);
  if (typeNameLit == "field")
    return generatedTypeParser(dialect->getContext(), parser, typeNameLit);
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
  if (typeNameLit == "shape")
    return generatedTypeParser(dialect->getContext(), parser, typeNameLit);
  if (typeNameLit == "shapeshift")
    return generatedTypeParser(dialect->getContext(), parser, typeNameLit);
  if (typeNameLit == "slice")
    return parseSlice(parser);
  if (typeNameLit == "tdesc")
    return parseTypeDesc(parser, loc);
  if (typeNameLit == "type")
    return parseDerived(parser, loc);
  if (typeNameLit == "void")
    return parseVoid(parser);
  if (typeNameLit == "vector")
    return parseVector(parser, loc);

  parser.emitError(parser.getNameLoc(), "unknown FIR type " + typeNameLit);
  return {};
}

namespace fir {
namespace detail {

// Type storage classes

/// `CHARACTER` storage
struct CharacterTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<KindTy, CharacterType::LenType>;

  static unsigned hashKey(const KeyTy &key) {
    auto hashVal = llvm::hash_combine(std::get<0>(key));
    return llvm::hash_combine(hashVal, llvm::hash_combine(std::get<1>(key)));
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy{getFKind(), getLen()};
  }

  static CharacterTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                         const KeyTy &key) {
    auto *storage = allocator.allocate<CharacterTypeStorage>();
    return new (storage)
        CharacterTypeStorage{std::get<0>(key), std::get<1>(key)};
  }

  KindTy getFKind() const { return kind; }
  CharacterType::LenType getLen() const { return len; }

protected:
  KindTy kind;
  CharacterType::LenType len;

private:
  CharacterTypeStorage() = delete;
  explicit CharacterTypeStorage(KindTy kind, CharacterType::LenType len)
      : kind{kind}, len{len} {}
};

struct SliceTypeStorage : public mlir::TypeStorage {
  using KeyTy = unsigned;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getRank(); }

  static SliceTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     unsigned rank) {
    auto *storage = allocator.allocate<SliceTypeStorage>();
    return new (storage) SliceTypeStorage{rank};
  }

  unsigned getRank() const { return rank; }

protected:
  unsigned rank;

private:
  SliceTypeStorage() = delete;
  explicit SliceTypeStorage(unsigned rank) : rank{rank} {}
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
struct IntegerTypeStorage : public mlir::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static IntegerTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       KindTy kind) {
    auto *storage = allocator.allocate<IntegerTypeStorage>();
    return new (storage) IntegerTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  IntegerTypeStorage() = delete;
  explicit IntegerTypeStorage(KindTy kind) : kind{kind} {}
};

/// `COMPLEX` storage
struct ComplexTypeStorage : public mlir::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static ComplexTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       KindTy kind) {
    auto *storage = allocator.allocate<ComplexTypeStorage>();
    return new (storage) ComplexTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  ComplexTypeStorage() = delete;
  explicit ComplexTypeStorage(KindTy kind) : kind{kind} {}
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

  // a !fir.boxchar<k> always wraps a !fir.char<k, ?>
  CharacterType getElementType(mlir::MLIRContext *ctxt) const {
    return CharacterType::getUnknownLen(ctxt, getFKind());
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
    auto shapeHash = hash_value(std::get<SequenceType::Shape>(key));
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

/// Vector type storage
struct VectorTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<uint64_t, mlir::Type>;

  static unsigned hashKey(const KeyTy &key) {
    return llvm::hash_combine(std::get<uint64_t>(key),
                              std::get<mlir::Type>(key));
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy{getLen(), getEleTy()};
  }

  static VectorTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    auto *storage = allocator.allocate<VectorTypeStorage>();
    return new (storage)
        VectorTypeStorage{std::get<uint64_t>(key), std::get<mlir::Type>(key)};
  }

  uint64_t getLen() const { return len; }
  mlir::Type getEleTy() const { return eleTy; }

protected:
  uint64_t len;
  mlir::Type eleTy;

private:
  VectorTypeStorage() = delete;
  explicit VectorTypeStorage(uint64_t len, mlir::Type eleTy)
      : len{len}, eleTy{eleTy} {}
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
  return t.getDialect().getNamespace().empty();
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

} // namespace fir

// CHARACTER

CharacterType fir::CharacterType::get(mlir::MLIRContext *ctxt, KindTy kind,
                                      CharacterType::LenType len) {
  return Base::get(ctxt, kind, len);
}

KindTy fir::CharacterType::getFKind() const { return getImpl()->getFKind(); }

CharacterType::LenType fir::CharacterType::getLen() const {
  return getImpl()->getLen();
}

// Len

LenType fir::LenType::get(mlir::MLIRContext *ctxt) {
  return Base::get(ctxt, 0);
}

// LOGICAL

LogicalType fir::LogicalType::get(mlir::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, kind);
}

KindTy fir::LogicalType::getFKind() const { return getImpl()->getFKind(); }

// INTEGER

fir::IntegerType fir::IntegerType::get(mlir::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, kind);
}

KindTy fir::IntegerType::getFKind() const { return getImpl()->getFKind(); }

// COMPLEX

fir::ComplexType fir::ComplexType::get(mlir::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, kind);
}

mlir::Type fir::ComplexType::getElementType() const {
  return fir::RealType::get(getContext(), getFKind());
}

KindTy fir::ComplexType::getFKind() const { return getImpl()->getFKind(); }

// REAL

RealType fir::RealType::get(mlir::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, kind);
}

KindTy fir::RealType::getFKind() const { return getImpl()->getFKind(); }

mlir::LogicalResult
fir::BoxType::verifyConstructionInvariants(mlir::Location, mlir::Type eleTy,
                                           mlir::AffineMapAttr map) {
  // TODO
  return mlir::success();
}

// BoxChar<C>

BoxCharType fir::BoxCharType::get(mlir::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, kind);
}

CharacterType fir::BoxCharType::getEleTy() const {
  return getImpl()->getElementType(getContext());
}

// BoxProc<T>

BoxProcType fir::BoxProcType::get(mlir::Type elementType) {
  return Base::get(elementType.getContext(), elementType);
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
  return Base::get(elementType.getContext(), elementType);
}

mlir::Type fir::ReferenceType::getEleTy() const {
  return getImpl()->getElementType();
}

mlir::LogicalResult
fir::ReferenceType::verifyConstructionInvariants(mlir::Location loc,
                                                 mlir::Type eleTy) {
  if (eleTy.isa<ShapeType>() || eleTy.isa<ShapeShiftType>() ||
      eleTy.isa<SliceType>() || eleTy.isa<FieldType>() ||
      eleTy.isa<LenType>() || eleTy.isa<ReferenceType>() ||
      eleTy.isa<TypeDescType>())
    return mlir::emitError(loc, "cannot build a reference to type: ")
           << eleTy << '\n';
  return mlir::success();
}

// Pointer<T>

PointerType fir::PointerType::get(mlir::Type elementType) {
  assert(singleIndirectionLevel(elementType) && "invalid element type");
  return Base::get(elementType.getContext(), elementType);
}

mlir::Type fir::PointerType::getEleTy() const {
  return getImpl()->getElementType();
}

static bool canBePointerOrHeapElementType(mlir::Type eleTy) {
  return eleTy.isa<BoxType>() || eleTy.isa<BoxCharType>() ||
         eleTy.isa<BoxProcType>() || eleTy.isa<ShapeType>() ||
         eleTy.isa<ShapeShiftType>() || eleTy.isa<SliceType>() ||
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
  return Base::get(elementType.getContext(), elementType);
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
  return Base::get(ctxt, shape, elementType, map);
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
      eleTy.isa<BoxProcType>() || eleTy.isa<ShapeType>() ||
      eleTy.isa<ShapeShiftType>() || eleTy.isa<SliceType>() ||
      eleTy.isa<FieldType>() || eleTy.isa<LenType>() || eleTy.isa<HeapType>() ||
      eleTy.isa<PointerType>() || eleTy.isa<ReferenceType>() ||
      eleTy.isa<TypeDescType>() || eleTy.isa<fir::VectorType>() ||
      eleTy.isa<SequenceType>())
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

// Slice

SliceType fir::SliceType::get(mlir::MLIRContext *ctxt, unsigned rank) {
  return Base::get(ctxt, rank);
}

unsigned fir::SliceType::getRank() const { return getImpl()->getRank(); }

/// RecordType
///
/// This type captures a Fortran "derived type"

RecordType fir::RecordType::get(mlir::MLIRContext *ctxt, llvm::StringRef name) {
  return Base::get(ctxt, name);
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

//===----------------------------------------------------------------------===//
// Type descriptor type
//===----------------------------------------------------------------------===//

TypeDescType fir::TypeDescType::get(mlir::Type ofType) {
  assert(!ofType.isa<ReferenceType>());
  return Base::get(ofType.getContext(), ofType);
}

mlir::Type fir::TypeDescType::getOfTy() const { return getImpl()->getOfType(); }

mlir::LogicalResult
fir::TypeDescType::verifyConstructionInvariants(mlir::Location loc,
                                                mlir::Type eleTy) {
  if (eleTy.isa<BoxType>() || eleTy.isa<BoxCharType>() ||
      eleTy.isa<BoxProcType>() || eleTy.isa<ShapeType>() ||
      eleTy.isa<ShapeShiftType>() || eleTy.isa<SliceType>() ||
      eleTy.isa<FieldType>() || eleTy.isa<LenType>() ||
      eleTy.isa<ReferenceType>() || eleTy.isa<TypeDescType>())
    return mlir::emitError(loc, "cannot build a type descriptor of type: ")
           << eleTy << '\n';
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Vector type
//===----------------------------------------------------------------------===//

fir::VectorType fir::VectorType::get(uint64_t len, mlir::Type eleTy) {
  return Base::get(eleTy.getContext(), len, eleTy);
}

mlir::Type fir::VectorType::getEleTy() const { return getImpl()->getEleTy(); }

uint64_t fir::VectorType::getLen() const { return getImpl()->getLen(); }

mlir::LogicalResult
fir::VectorType::verifyConstructionInvariants(mlir::Location loc, uint64_t len,
                                              mlir::Type eleTy) {
  if (!(fir::isa_real(eleTy) || fir::isa_integer(eleTy)))
    return mlir::emitError(loc, "cannot build a vector of type ")
           << eleTy << '\n';
  return mlir::success();
}

namespace {

void printBounds(llvm::raw_ostream &os, const SequenceType::Shape &bounds) {
  os << '<';
  for (auto &b : bounds) {
    if (b >= 0)
      os << b << 'x';
    else
      os << "?x";
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
  if (auto type = ty.dyn_cast<BoxCharType>()) {
    os << "boxchar<" << type.getEleTy().cast<fir::CharacterType>().getFKind()
       << '>';
    return;
  }
  if (auto type = ty.dyn_cast<BoxProcType>()) {
    os << "boxproc<";
    p.printType(type.getEleTy());
    os << '>';
    return;
  }
  if (auto chTy = ty.dyn_cast<CharacterType>()) {
    // Fortran intrinsic type CHARACTER
    os << "char<" << chTy.getFKind();
    auto len = chTy.getLen();
    if (len != fir::CharacterType::singleton()) {
      os << ',';
      if (len == fir::CharacterType::unknownLen())
        os << '?';
      else
        os << len;
    }
    os << '>';
    return;
  }
  if (auto type = ty.dyn_cast<fir::ComplexType>()) {
    // Fortran intrinsic type COMPLEX
    os << "complex<" << type.getFKind() << '>';
    return;
  }
  if (auto type = ty.dyn_cast<RecordType>()) {
    // Fortran derived type
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
    return;
  }
  if (auto type = ty.dyn_cast<SliceType>()) {
    os << "slice<" << type.getRank() << '>';
    return;
  }
  if (auto type = ty.dyn_cast<HeapType>()) {
    os << "heap<";
    p.printType(type.getEleTy());
    os << '>';
    return;
  }
  if (auto type = ty.dyn_cast<fir::IntegerType>()) {
    // Fortran intrinsic type INTEGER
    os << "int<" << type.getFKind() << '>';
    return;
  }
  if (auto type = ty.dyn_cast<LenType>()) {
    os << "len";
    return;
  }
  if (auto type = ty.dyn_cast<LogicalType>()) {
    // Fortran intrinsic type LOGICAL
    os << "logical<" << type.getFKind() << '>';
    return;
  }
  if (auto type = ty.dyn_cast<PointerType>()) {
    os << "ptr<";
    p.printType(type.getEleTy());
    os << '>';
    return;
  }
  if (auto type = ty.dyn_cast<fir::RealType>()) {
    // Fortran intrinsic types REAL and DOUBLE PRECISION
    os << "real<" << type.getFKind() << '>';
    return;
  }
  if (auto type = ty.dyn_cast<ReferenceType>()) {
    os << "ref<";
    p.printType(type.getEleTy());
    os << '>';
    return;
  }
  if (auto type = ty.dyn_cast<SequenceType>()) {
    os << "array";
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
    return;
  }
  if (auto type = ty.dyn_cast<TypeDescType>()) {
    os << "tdesc<";
    p.printType(type.getOfTy());
    os << '>';
    return;
  }
  if (auto type = ty.dyn_cast<fir::VectorType>()) {
    os << "vector<" << type.getLen() << ':';
    p.printType(type.getEleTy());
    os << '>';
    return;
  }

  if (mlir::succeeded(generatedTypePrinter(ty, p))) {
    return;
  }
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

namespace fir {

//===----------------------------------------------------------------------===//
// BoxType
//===----------------------------------------------------------------------===//

// `box` `<` type (',' affine-map)? `>`
mlir::Type BoxType::parse(mlir::MLIRContext *context,
                          mlir::DialectAsmParser &parser) {
  mlir::Type ofTy;
  if (parser.parseLess() || parser.parseType(ofTy)) {
    parser.emitError(parser.getCurrentLocation(), "expected type parameter");
    return Type();
  }

  mlir::AffineMapAttr map;
  if (!parser.parseOptionalComma()) {
    if (parser.parseAttribute(map)) {
      parser.emitError(parser.getCurrentLocation(), "expected affine map");
      return Type();
    }
  }
  if (parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation(), "expected '>'");
    return Type();
  }
  return get(ofTy, map);
}

void BoxType::print(::mlir::DialectAsmPrinter &printer) const {
  printer << "box<";
  printer.printType(getEleTy());
  if (auto map = getLayoutMap()) {
    printer << ", ";
    printer.printAttribute(map);
  }
  printer << '>';
}

} // namespace fir
