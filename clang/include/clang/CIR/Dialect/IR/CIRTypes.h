//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRTYPES_H
#define CLANG_CIR_DIALECT_IR_CIRTYPES_H

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Interfaces/CIRTypeInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
struct fltSemantics;
} // namespace llvm

namespace cir {

namespace detail {
struct StructTypeStorage;
struct UnionTypeStorage;
} // namespace detail

bool isValidFundamentalIntWidth(unsigned width);

/// Whether a member of this kind holds data for argument passing.  An `empty`
/// member can hold data all the same, since an access unit of unnamed
/// bit-fields takes that mark and is storage the classifier reads; use
/// RecordType::isEmptyForABI to ask about a whole record.
inline bool holdsDataForABI(RecordMemberKind kind) {
  return kind == RecordMemberKind::Data || kind == RecordMemberKind::BitField;
}

/// Whether any member holds data for argument passing on its mark alone.
inline bool anyMemberHoldsDataForABI(llvm::ArrayRef<RecordMemberKind> kinds) {
  return llvm::any_of(
      kinds, [](RecordMemberKind kind) { return holdsDataForABI(kind); });
}

/// Whether a member of this kind is an access unit the source can read a
/// bit-field of.  A unit can be narrower than the type a bit-field in it was
/// declared with, so its extent does not answer what the source declared;
/// `!cir.bitfield` carries that.  A true answer does not mean the member is an
/// access unit: a union's base subobject takes this mark when any variant is
/// one, whatever its own storage type came from.  A unit of nothing but
/// unnamed bit-fields is `empty` instead.
inline bool isNamedBitField(RecordMemberKind kind) {
  return kind == RecordMemberKind::BitField;
}

/// Returns true if the type is a CIR sized type.
///
/// Types are sized if they implement SizedTypeInterface and
/// return true from its method isSized.
///
/// Unsized types are those that do not have a size, such as
/// void, or abstract types.
bool isSized(mlir::Type ty);

/// Returns the CIR floating-point type for the given semantics, or a null
/// type if CIR has no type for it (e.g. PPCDoubleDouble or a Float8 format).
/// Mirrors llvm::Type::getFloatingPointTy.
cir::FPTypeInterface getFloatingPointType(const llvm::fltSemantics &sem,
                                          mlir::MLIRContext *ctx);

//===----------------------------------------------------------------------===//
// AddressSpace helpers
//===----------------------------------------------------------------------===//

cir::LangAddressSpace toCIRLangAddressSpace(clang::LangAS langAS);

// Compare a CIR memory space attribute with a Clang LangAS.
bool isMatchingAddressSpace(mlir::ptr::MemorySpaceAttrInterface cirAS,
                            clang::LangAS as);

/// Convert an AST LangAS to the appropriate CIR address space attribute
/// interface.
mlir::ptr::MemorySpaceAttrInterface
toCIRAddressSpaceAttr(mlir::MLIRContext &ctx, clang::LangAS langAS);

/// Normalize LangAddressSpace::Default to null (empty attribute).
mlir::ptr::MemorySpaceAttrInterface
normalizeDefaultAddressSpace(mlir::ptr::MemorySpaceAttrInterface addrSpace);

bool isSupportedCIRMemorySpaceAttr(
    mlir::ptr::MemorySpaceAttrInterface memorySpace);

} // namespace cir

//===----------------------------------------------------------------------===//
// CIR Dialect Tablegen'd Types
//===----------------------------------------------------------------------===//

namespace cir {

#include "clang/CIR/Dialect/IR/CIRTypeConstraints.h.inc"

} // namespace cir

#define GET_TYPEDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsTypes.h.inc"

namespace cir {

/// Whether a record member occupies bytes of its record.  Every member does
/// except a zero-width bit-field, which contributes neither size nor alignment
/// to the record and is left out of the lowered LLVM struct body.
inline bool memberOwnsBytes(mlir::Type memberTy) {
  if (auto bfTy = mlir::dyn_cast<cir::BitFieldType>(memberTy))
    return bfTy.ownsBytes();
  return true;
}

/// The storage a member is stored as: the access unit for a bit-field member,
/// and the member type itself for anything else.  Null for a zero-width
/// bit-field, which has no storage.
inline mlir::Type memberStorageType(mlir::Type memberTy) {
  if (auto bfTy = mlir::dyn_cast<cir::BitFieldType>(memberTy))
    return bfTy.getStorageType();
  return memberTy;
}

/// C++ view class that accepts both !cir.struct and !cir.union types.
///
/// Follows the MLIR BaseMemRefType pattern: StructType and UnionType are the
/// concrete tablegen types; RecordType is a hand-written view class that
/// covers both.  Use it when code must handle either kind generically.
///
/// Methods that are common to both types are forwarded through dyn_cast
/// dispatch.  Type-specific methods (getPadding, getUnionStorageType) are only
/// available on the concrete type.
class RecordType : public mlir::Type {
public:
  using mlir::Type::Type;

  // Allow implicit construction from concrete record types so that
  // functions returning cir::RecordType can return StructType/UnionType
  // values without an explicit cast.
  // NOLINTNEXTLINE(google-explicit-constructor)
  RecordType(StructType t) : mlir::Type(t) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  RecordType(UnionType t) : mlir::Type(t) {}

  static bool classof(mlir::Type t) {
    return mlir::isa<StructType>(t) || mlir::isa<UnionType>(t);
  }

  llvm::ArrayRef<mlir::Type> getMembers() const;
  mlir::StringAttr getName() const;
  bool isIncomplete() const;
  bool isComplete() const { return !isIncomplete(); }
  bool getPacked() const;
  bool getPadded() const;
  llvm::ArrayRef<RecordMemberKind> getMemberKinds() const;

  bool isClass() const;
  bool isStruct() const;
  bool isUnion() const { return mlir::isa<UnionType>(*this); }

  /// Whether no member holds data.  Vacuously true for a complete record with
  /// no members, and false for an incomplete one, whose members are not known
  /// yet.  A union's tail-padding slot is not a member and does not count.
  bool isEmptyForABI() const;

  /// One `Data` kind per member.  Takes the member list rather than a count so
  /// the length cannot drift from the record it describes.
  static llvm::SmallVector<RecordMemberKind>
  getAllDataKinds(llvm::ArrayRef<mlir::Type> members);

  size_t getNumElements() const { return getMembers().size(); }
  mlir::Type getElementType(size_t idx) const { return getMembers()[idx]; }
  std::string getKindAsStr() const;
  std::string getPrefixedName() const;

  /// \p padding is union-only.  A struct carries its padding as a member
  /// marked pad.
  void complete(llvm::ArrayRef<mlir::Type> members, bool packed,
                mlir::Type padding,
                llvm::ArrayRef<RecordMemberKind> memberKinds);
  uint64_t getElementOffset(const mlir::DataLayout &dataLayout,
                            unsigned idx) const;
  bool isLayoutIdentical(const RecordType &other);

  bool isABIConvertedRecord() const;
  mlir::StringAttr getABIConvertedName() const;
  void removeABIConversionNamePrefix();
};

} // namespace cir

#endif // CLANG_CIR_DIALECT_IR_CIRTYPES_H
