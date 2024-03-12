//===--- CIRGenRecordLayout.h - CIR Record Layout Information ---*- C++ -*-===//
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

#include "llvm/Support/raw_ostream.h"

namespace cir {

/// Structure with information about how a bitfield should be accessed. This is
/// very similar to what LLVM codegen does, once CIR evolves it's possible we
/// can use a more higher level representation.
/// TODO(cir): the comment below is extracted from LLVM, build a CIR version of
/// this.
///
/// Often we layout a sequence of bitfields as a contiguous sequence of bits.
/// When the AST record layout does this, we represent it in the LLVM IR's type
/// as either a sequence of i8 members or a byte array to reserve the number of
/// bytes touched without forcing any particular alignment beyond the basic
/// character alignment.
///
/// Then accessing a particular bitfield involves converting this byte array
/// into a single integer of that size (i24 or i40 -- may not be power-of-two
/// size), loading it, and shifting and masking to extract the particular
/// subsequence of bits which make up that particular bitfield. This structure
/// encodes the information used to construct the extraction code sequences.
/// The CIRGenRecordLayout also has a field index which encodes which
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
/// This will end up as the following LLVM type. The first array is the
/// bitfield, and the second is the padding out to a 4-byte alignment.
///
///   %t = type { i8, i8, i8, i8, i8, [3 x i8] }
///
/// When generating code to access more_bits, we'll generate something
/// essentially like this:
///
///   define i32 @foo(%t* %base) {
///     %0 = gep %t* %base, i32 0, i32 3
///     %2 = load i8* %1
///     %3 = lshr i8 %2, 3
///     %4 = and i8 %3, 15
///     %5 = zext i8 %4 to i32
///     ret i32 %i
///   }
///
struct CIRGenBitFieldInfo {
  /// The offset within a contiguous run of bitfields that are represented as
  /// a single "field" within the LLVM struct type. This offset is in bits.
  unsigned Offset : 16;

  /// The total size of the bit-field, in bits.
  unsigned Size : 15;

  /// Whether the bit-field is signed.
  unsigned IsSigned : 1;

  /// The storage size in bits which should be used when accessing this
  /// bitfield.
  unsigned StorageSize;

  /// The offset of the bitfield storage from the start of the struct.
  clang::CharUnits StorageOffset;

  /// The offset within a contiguous run of bitfields that are represented as a
  /// single "field" within the LLVM struct type, taking into account the AAPCS
  /// rules for volatile bitfields. This offset is in bits.
  unsigned VolatileOffset : 16;

  /// The storage size in bits which should be used when accessing this
  /// bitfield.
  unsigned VolatileStorageSize;

  /// The offset of the bitfield storage from the start of the struct.
  clang::CharUnits VolatileStorageOffset;

  /// The name of a bitfield
  llvm::StringRef Name;

  // The actual storage type for the bitfield
  mlir::Type StorageType;

  CIRGenBitFieldInfo()
      : Offset(), Size(), IsSigned(), StorageSize(), VolatileOffset(),
        VolatileStorageSize() {}

  CIRGenBitFieldInfo(unsigned Offset, unsigned Size, bool IsSigned,
                     unsigned StorageSize, clang::CharUnits StorageOffset)
      : Offset(Offset), Size(Size), IsSigned(IsSigned),
        StorageSize(StorageSize), StorageOffset(StorageOffset) {}

  void print(llvm::raw_ostream &OS) const;
  void dump() const;

  /// Given a bit-field decl, build an appropriate helper object for
  /// accessing that field (which is expected to have the given offset and
  /// size).
  static CIRGenBitFieldInfo MakeInfo(class CIRGenTypes &Types,
                                     const clang::FieldDecl *FD,
                                     uint64_t Offset, uint64_t Size,
                                     uint64_t StorageSize,
                                     clang::CharUnits StorageOffset);
};

/// This class handles struct and union layout info while lowering AST types
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
  mlir::cir::StructType CompleteObjectType;

  /// The CIR type for the non-virtual part of this record layout; used when
  /// laying it out as a base subobject.
  mlir::cir::StructType BaseSubobjectType;

  /// Map from (non-bit-field) struct field to the corresponding cir struct type
  /// field no. This info is populated by the record builder.
  llvm::DenseMap<const clang::FieldDecl *, unsigned> FieldInfo;

  /// Map from (bit-field) struct field to the corresponding CIR struct type
  /// field no. This info is populated by record builder.
  /// TODO(CIR): value is an int for now, fix when we support bitfields
  llvm::DenseMap<const clang::FieldDecl *, CIRGenBitFieldInfo> BitFields;

  // FIXME: Maybe we could use CXXBaseSpecifier as the key and use a single map
  // for both virtual and non-virtual bases.
  llvm::DenseMap<const clang::CXXRecordDecl *, unsigned> NonVirtualBases;

  /// Map from virtual bases to their field index in the complete object.
  llvm::DenseMap<const clang::CXXRecordDecl *, unsigned>
      CompleteObjectVirtualBases;

  /// False if any direct or indirect subobject of this class, when considered
  /// as a complete object, requires a non-zero bitpattern when
  /// zero-initialized.
  bool IsZeroInitializable : 1;

  /// False if any direct or indirect subobject of this class, when considered
  /// as a base subobject, requires a non-zero bitpattern when zero-initialized.
  bool IsZeroInitializableAsBase : 1;

public:
  CIRGenRecordLayout(mlir::cir::StructType CompleteObjectType,
                     mlir::cir::StructType BaseSubobjectType,
                     bool IsZeroInitializable, bool IsZeroInitializableAsBase)
      : CompleteObjectType(CompleteObjectType),
        BaseSubobjectType(BaseSubobjectType),
        IsZeroInitializable(IsZeroInitializable),
        IsZeroInitializableAsBase(IsZeroInitializableAsBase) {}

  /// Return the "complete object" LLVM type associated with
  /// this record.
  mlir::cir::StructType getCIRType() const { return CompleteObjectType; }

  /// Return the "base subobject" LLVM type associated with
  /// this record.
  mlir::cir::StructType getBaseSubobjectCIRType() const {
    return BaseSubobjectType;
  }

  /// Return cir::StructType element number that corresponds to the field FD.
  unsigned getCIRFieldNo(const clang::FieldDecl *FD) const {
    FD = FD->getCanonicalDecl();
    assert(FieldInfo.count(FD) && "Invalid field for record!");
    return FieldInfo.lookup(FD);
  }

  /// Check whether this struct can be C++ zero-initialized with a
  /// zeroinitializer.
  bool isZeroInitializable() const { return IsZeroInitializable; }

  /// Return the BitFieldInfo that corresponds to the field FD.
  const CIRGenBitFieldInfo &getBitFieldInfo(const clang::FieldDecl *FD) const {
    FD = FD->getCanonicalDecl();
    assert(FD->isBitField() && "Invalid call for non-bit-field decl!");
    llvm::DenseMap<const clang::FieldDecl *, CIRGenBitFieldInfo>::const_iterator
        it = BitFields.find(FD);
    assert(it != BitFields.end() && "Unable to find bitfield info");
    return it->second;
  }
};

} // namespace cir

#endif
