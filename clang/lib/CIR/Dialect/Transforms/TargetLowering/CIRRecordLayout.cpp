//===- CIRRecordLayout.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/AST/RecordLayout.cpp. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "CIRRecordLayout.h"
#include "clang/CIR/MissingFeatures.h"

namespace mlir {
namespace cir {

// Constructor for C++ records.
CIRRecordLayout::CIRRecordLayout(
    const CIRLowerContext &Ctx, clang::CharUnits size,
    clang::CharUnits alignment, clang::CharUnits preferredAlignment,
    clang::CharUnits unadjustedAlignment, clang::CharUnits requiredAlignment,
    bool hasOwnVFPtr, bool hasExtendableVFPtr, clang::CharUnits vbptroffset,
    clang::CharUnits datasize, ArrayRef<uint64_t> fieldoffsets,
    clang::CharUnits nonvirtualsize, clang::CharUnits nonvirtualalignment,
    clang::CharUnits preferrednvalignment,
    clang::CharUnits SizeOfLargestEmptySubobject, const Type PrimaryBase,
    bool IsPrimaryBaseVirtual, const Type BaseSharingVBPtr,
    bool EndsWithZeroSizedObject, bool LeadsWithZeroSizedBase)
    : Size(size), DataSize(datasize), Alignment(alignment),
      PreferredAlignment(preferredAlignment),
      UnadjustedAlignment(unadjustedAlignment),
      RequiredAlignment(requiredAlignment), CXXInfo(new CXXRecordLayoutInfo) {
  // NOTE(cir): Clang does a far more elaborate append here by leveraging the
  // custom ASTVector class. For now, we'll do a simple append.
  FieldOffsets.insert(FieldOffsets.end(), fieldoffsets.begin(),
                      fieldoffsets.end());

  cir_tl_assert(!PrimaryBase && "Layout for class with inheritance is NYI");
  // CXXInfo->PrimaryBase.setPointer(PrimaryBase);
  cir_tl_assert(!IsPrimaryBaseVirtual && "Layout for virtual base class is NYI");
  // CXXInfo->PrimaryBase.setInt(IsPrimaryBaseVirtual);
  CXXInfo->NonVirtualSize = nonvirtualsize;
  CXXInfo->NonVirtualAlignment = nonvirtualalignment;
  CXXInfo->PreferredNVAlignment = preferrednvalignment;
  CXXInfo->SizeOfLargestEmptySubobject = SizeOfLargestEmptySubobject;
  // FIXME(cir): Initialize base classes offsets.
  cir_tl_assert(!::cir::MissingFeatures::getCXXRecordBases());
  CXXInfo->HasOwnVFPtr = hasOwnVFPtr;
  CXXInfo->VBPtrOffset = vbptroffset;
  CXXInfo->HasExtendableVFPtr = hasExtendableVFPtr;
  // FIXME(cir): Probably not necessary for now.
  // CXXInfo->BaseSharingVBPtr = BaseSharingVBPtr;
  CXXInfo->EndsWithZeroSizedObject = EndsWithZeroSizedObject;
  CXXInfo->LeadsWithZeroSizedBase = LeadsWithZeroSizedBase;
}

} // namespace cir
} // namespace mlir
