//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENRECORDLAYOUT_H
#define LLVM_CLANG_LIB_CIR_CIRGENRECORDLAYOUT_H

#include "clang/AST/Decl.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace clang::CIRGen {

/// Record with information about how a bitfield should be accessed. This is
/// very similar to what LLVM codegen does, once CIR evolves it's possible we
/// can use a more higher level representation.
///
/// Often we lay out a sequence of bitfields as a contiguous sequence of bits.
/// When the AST record layout does this, we represent it in CIR as a
/// `!cir.record` type, which directly reflects the structure's layout,
/// including bitfield packing and padding, using CIR types such as
/// `!cir.bool`, `!s8i`, `!u16i`.
///
/// To access a particular bitfield in CIR, we use the operations
/// `cir.get_bitfield` (`GetBitfieldOp`) or `cir.set_bitfield`
/// (`SetBitfieldOp`). These operations rely on the `bitfield_info`
/// attribute, which provides detailed metadata required for access,
/// such as the size and offset of the bitfield, the type and size of
/// the underlying storage, and whether the value is signed.
/// The CIRGenRecordLayout also has a bitFields map which encodes which
/// byte-sequence this bitfield falls within. Let's assume the following C
/// struct:
///
///   struct S {
///     char a, b, c;
///     unsigned bits : 3;
///     unsigned more_bits : 4;
///     unsigned still_more_bits : 7;
///   };
///
/// This will end up as the following cir.record. The bitfield members are
/// represented by one !u16i value, and the array provides padding to align the
/// struct to a 4-byte alignment.
///
///   !rec_S = !cir.record<struct "S" padded {!s8i, !s8i, !s8i, !u16i,
///   !cir.array<!u8i x 3>}>
///
/// When generating code to access more_bits, we'll generate something
/// essentially like this:
///
///   #bfi_more_bits = #cir.bitfield_info<name = "more_bits", storage_type =
///   !u16i, size = 4, offset = 3, is_signed = false>
///
///   cir.func @store_field() {
///     %0 = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s"] {alignment = 4 : i64}
///     %1 = cir.const #cir.int<2> : !s32i
///     %2 = cir.cast(integral, %1 : !s32i), !u32i
///     %3 = cir.get_member %0[3] {name = "more_bits"} : !cir.ptr<!rec_S> ->
///     !cir.ptr<!u16i>
///     %4 = cir.set_bitfield(#bfi_more_bits, %3 :
///     !cir.ptr<!u16i>, %2 : !u32i) -> !u32i
///     cir.return
///   }
///
struct CIRGenBitFieldInfo {
  /// The offset within a contiguous run of bitfields that are represented as
  /// a single "field" within the cir.record type. This offset is in bits.
  unsigned offset : 16;

  /// The total size of the bit-field, in bits.
  unsigned size : 15;

  /// Whether the bit-field is signed.
  unsigned isSigned : 1;

  /// The storage size in bits which should be used when accessing this
  /// bitfield.
  unsigned storageSize;

  /// The offset of the bitfield storage from the start of the record.
  clang::CharUnits storageOffset;

  /// The offset within a contiguous run of bitfields that are represented as a
  /// single "field" within the cir.record type, taking into account the AAPCS
  /// rules for volatile bitfields. This offset is in bits.
  unsigned volatileOffset : 16;

  /// The storage size in bits which should be used when accessing this
  /// bitfield.
  unsigned volatileStorageSize;

  /// The offset of the bitfield storage from the start of the record.
  clang::CharUnits volatileStorageOffset;

  /// The name of a bitfield
  llvm::StringRef name;

  // The actual storage type for the bitfield
  mlir::Type storageType;

  CIRGenBitFieldInfo()
      : offset(), size(), isSigned(), storageSize(), volatileOffset(),
        volatileStorageSize() {}

  CIRGenBitFieldInfo(unsigned offset, unsigned size, bool isSigned,
                     unsigned storageSize, clang::CharUnits storageOffset)
      : offset(offset), size(size), isSigned(isSigned),
        storageSize(storageSize), storageOffset(storageOffset) {}

  void print(llvm::raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

/// This class handles record and union layout info while lowering AST types
/// to CIR types.
///
/// These layout objects are only created on demand as CIR generation requires.
class CIRGenRecordLayout {
  friend class CIRGenTypes;

  CIRGenRecordLayout(const CIRGenRecordLayout &) = delete;
  void operator=(const CIRGenRecordLayout &) = delete;

private:
  /// The CIR type corresponding to this record layout; used when laying it out
  /// as a complete object.
  cir::RecordType completeObjectType;

  /// The CIR type for the non-virtual part of this record layout; used when
  /// laying it out as a base subobject.
  cir::RecordType baseSubobjectType;

  /// Map from (non-bit-field) record field to the corresponding cir record type
  /// field no. This info is populated by the record builder.
  llvm::DenseMap<const clang::FieldDecl *, unsigned> fieldIdxMap;

  // FIXME: Maybe we could use CXXBaseSpecifier as the key and use a single map
  // for both virtual and non-virtual bases.
  llvm::DenseMap<const clang::CXXRecordDecl *, unsigned> nonVirtualBases;

  /// Map from virtual bases to their field index in the complete object.
  llvm::DenseMap<const clang::CXXRecordDecl *, unsigned>
      completeObjectVirtualBases;

  /// Map from (bit-field) record field to the corresponding CIR record type
  /// field no. This info is populated by record builder.
  llvm::DenseMap<const clang::FieldDecl *, CIRGenBitFieldInfo> bitFields;

  /// False if any direct or indirect subobject of this class, when considered
  /// as a complete object, requires a non-zero bitpattern when
  /// zero-initialized.
  LLVM_PREFERRED_TYPE(bool)
  unsigned zeroInitializable : 1;

  /// False if any direct or indirect subobject of this class, when considered
  /// as a base subobject, requires a non-zero bitpattern when zero-initialized.
  LLVM_PREFERRED_TYPE(bool)
  unsigned zeroInitializableAsBase : 1;

public:
  CIRGenRecordLayout(cir::RecordType completeObjectType,
                     cir::RecordType baseSubobjectType, bool zeroInitializable,
                     bool zeroInitializableAsBase)
      : completeObjectType(completeObjectType),
        baseSubobjectType(baseSubobjectType),
        zeroInitializable(zeroInitializable),
        zeroInitializableAsBase(zeroInitializableAsBase) {}

  /// Return the "complete object" LLVM type associated with
  /// this record.
  cir::RecordType getCIRType() const { return completeObjectType; }

  /// Return the "base subobject" LLVM type associated with
  /// this record.
  cir::RecordType getBaseSubobjectCIRType() const { return baseSubobjectType; }

  /// Return cir::RecordType element number that corresponds to the field FD.
  unsigned getCIRFieldNo(const clang::FieldDecl *fd) const {
    fd = fd->getCanonicalDecl();
    assert(fieldIdxMap.count(fd) && "Invalid field for record!");
    return fieldIdxMap.lookup(fd);
  }

  /// Check whether this struct can be C++ zero-initialized
  /// with a zeroinitializer.
  bool isZeroInitializable() const { return zeroInitializable; }

  /// Check whether this struct can be C++ zero-initialized
  /// with a zeroinitializer when considered as a base subobject.
  bool isZeroInitializableAsBase() const { return zeroInitializableAsBase; }

  /// Return the BitFieldInfo that corresponds to the field FD.
  const CIRGenBitFieldInfo &getBitFieldInfo(const clang::FieldDecl *fd) const {
    fd = fd->getCanonicalDecl();
    assert(fd->isBitField() && "Invalid call for non-bit-field decl!");
    llvm::DenseMap<const clang::FieldDecl *, CIRGenBitFieldInfo>::const_iterator
        it = bitFields.find(fd);
    assert(it != bitFields.end() && "Unable to find bitfield info");
    return it->second;
  }
  void print(raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

} // namespace clang::CIRGen

#endif
