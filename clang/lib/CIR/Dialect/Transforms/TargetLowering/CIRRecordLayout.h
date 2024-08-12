//===--- CGRecordLayout.h - LLVM Record Layout Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CGRecordLayout.h. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRRECORDLAYOUT_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRRECORDLAYOUT_H

#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "clang/AST/CharUnits.h"
#include <cstdint>
#include <vector>

namespace mlir {
namespace cir {

class CIRLowerContext;

// FIXME(cir): Perhaps this logic can be moved to the CIR dialect, specifically
// the data layout abstractions.

/// This class contains layout information for one RecordDecl, which is a
/// struct/union/class.  The decl represented must be a definition, not a
/// forward declaration. This class is also used to contain layout information
/// for one ObjCInterfaceDecl.
class CIRRecordLayout {

private:
  friend class CIRLowerContext;

  /// Size of record in characters.
  clang::CharUnits Size;

  /// Size of record in characters without tail padding.
  clang::CharUnits DataSize;

  // Alignment of record in characters.
  clang::CharUnits Alignment;

  // Preferred alignment of record in characters. This can be different than
  // Alignment in cases where it is beneficial for performance or backwards
  // compatibility preserving (e.g. AIX-ABI).
  clang::CharUnits PreferredAlignment;

  // Maximum of the alignments of the record members in characters.
  clang::CharUnits UnadjustedAlignment;

  /// The required alignment of the object. In the MS-ABI the
  /// __declspec(align()) trumps #pramga pack and must always be obeyed.
  clang::CharUnits RequiredAlignment;

  /// Array of field offsets in bits.
  /// FIXME(cir): Create a custom CIRVector instead?
  std::vector<uint64_t> FieldOffsets;

  struct CXXRecordLayoutInfo {
    /// The non-virtual size (in chars) of an object, which is the size of the
    /// object without virtual bases.
    clang::CharUnits NonVirtualSize;

    /// The non-virtual alignment (in chars) of an object, which is the
    /// alignment of the object without virtual bases.
    clang::CharUnits NonVirtualAlignment;

    /// The preferred non-virtual alignment (in chars) of an object, which is
    /// the preferred alignment of the object without virtual bases.
    clang::CharUnits PreferredNVAlignment;

    /// The size of the largest empty subobject (either a base or a member).
    /// Will be zero if the class doesn't contain any empty subobjects.
    clang::CharUnits SizeOfLargestEmptySubobject;

    /// Virtual base table offset (Microsoft-only).
    clang::CharUnits VBPtrOffset;

    /// Does this class provide a virtual function table (vtable in Itanium,
    /// vftbl in Microsoft) that is independent from its base classes?
    bool HasOwnVFPtr : 1;

    /// Does this class have a vftable that could be extended by a derived
    /// class.  The class may have inherited this pointer from a primary base
    /// class.
    bool HasExtendableVFPtr : 1;

    /// True if this class contains a zero sized member or base or a base with a
    /// zero sized member or base. Only used for MS-ABI.
    bool EndsWithZeroSizedObject : 1;

    /// True if this class is zero sized or first base is zero sized or has this
    /// property.  Only used for MS-ABI.
    bool LeadsWithZeroSizedBase : 1;
  };

  /// CXXInfo - If the record layout is for a C++ record, this will have
  /// C++ specific information about the record.
  CXXRecordLayoutInfo *CXXInfo = nullptr;

  // Constructor for C++ records.
  CIRRecordLayout(
      const CIRLowerContext &Ctx, clang::CharUnits size,
      clang::CharUnits alignment, clang::CharUnits preferredAlignment,
      clang::CharUnits unadjustedAlignment, clang::CharUnits requiredAlignment,
      bool hasOwnVFPtr, bool hasExtendableVFPtr, clang::CharUnits vbptroffset,
      clang::CharUnits datasize, ArrayRef<uint64_t> fieldoffsets,
      clang::CharUnits nonvirtualsize, clang::CharUnits nonvirtualalignment,
      clang::CharUnits preferrednvalignment,
      clang::CharUnits SizeOfLargestEmptySubobject, const Type PrimaryBase,
      bool IsPrimaryBaseVirtual, const Type BaseSharingVBPtr,
      bool EndsWithZeroSizedObject, bool LeadsWithZeroSizedBase);

  ~CIRRecordLayout() = default;

public:
  /// Get the record alignment in characters.
  clang::CharUnits getAlignment() const { return Alignment; }

  /// Get the record size in characters.
  clang::CharUnits getSize() const { return Size; }

  /// Get the offset of the given field index, in bits.
  uint64_t getFieldOffset(unsigned FieldNo) const {
    return FieldOffsets[FieldNo];
  }
};

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRRECORDLAYOUT_H
