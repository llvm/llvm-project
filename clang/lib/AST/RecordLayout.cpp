//===-- RecordLayout.cpp - Layout information for a struct/union -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the RecordLayout interface.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/TargetInfo.h"

using namespace clang;

void ASTRecordLayout::Destroy(ASTContext &Ctx) {
  if (FieldOffsets)
    Ctx.Deallocate(FieldOffsets);
  if (CXXInfo) {
    Ctx.Deallocate(CXXInfo);
    CXXInfo->~CXXRecordLayoutInfo();
  }
  this->~ASTRecordLayout();
  Ctx.Deallocate(this);
}

ASTRecordLayout::ASTRecordLayout(const ASTContext &Ctx, CharUnits size,
                                 CharUnits alignment, CharUnits datasize,
                                 const uint64_t *fieldoffsets,
                                 unsigned fieldcount)
  : Size(size), DataSize(datasize), Alignment(alignment), FieldOffsets(0),
    FieldCount(fieldcount), CXXInfo(0) {
  if (FieldCount > 0)  {
    FieldOffsets = new (Ctx) uint64_t[FieldCount];
    memcpy(FieldOffsets, fieldoffsets, FieldCount * sizeof(*FieldOffsets));
  }
}

// Constructor for C++ records.
ASTRecordLayout::ASTRecordLayout(const ASTContext &Ctx,
                                 CharUnits size, CharUnits alignment,
                                 bool hasOwnVFPtr, CharUnits vbptroffset,
                                 CharUnits datasize,
                                 const uint64_t *fieldoffsets,
                                 unsigned fieldcount,
                                 CharUnits nonvirtualsize,
                                 CharUnits nonvirtualalign,
                                 CharUnits SizeOfLargestEmptySubobject,
                                 const CXXRecordDecl *PrimaryBase,
                                 bool IsPrimaryBaseVirtual,
                                 const BaseOffsetsMapTy& BaseOffsets,
                                 const VBaseOffsetsMapTy& VBaseOffsets)
  : Size(size), DataSize(datasize), Alignment(alignment), FieldOffsets(0),
    FieldCount(fieldcount), CXXInfo(new (Ctx) CXXRecordLayoutInfo)
{
  if (FieldCount > 0)  {
    FieldOffsets = new (Ctx) uint64_t[FieldCount];
    memcpy(FieldOffsets, fieldoffsets, FieldCount * sizeof(*FieldOffsets));
  }

  CXXInfo->PrimaryBase.setPointer(PrimaryBase);
  CXXInfo->PrimaryBase.setInt(IsPrimaryBaseVirtual);
  CXXInfo->NonVirtualSize = nonvirtualsize;
  CXXInfo->NonVirtualAlign = nonvirtualalign;
  CXXInfo->SizeOfLargestEmptySubobject = SizeOfLargestEmptySubobject;
  CXXInfo->BaseOffsets = BaseOffsets;
  CXXInfo->VBaseOffsets = VBaseOffsets;
  CXXInfo->HasOwnVFPtr = hasOwnVFPtr;
  CXXInfo->VBPtrOffset = vbptroffset;

#ifndef NDEBUG
    if (const CXXRecordDecl *PrimaryBase = getPrimaryBase()) {
      if (isPrimaryBaseVirtual()) {
        // Microsoft ABI doesn't have primary virtual base
        if (Ctx.getTargetInfo().getCXXABI() != CXXABI_Microsoft) {
        assert(getVBaseClassOffset(PrimaryBase).isZero() &&
               "Primary virtual base must be at offset 0!");
        }
      } else {
        assert(getBaseClassOffset(PrimaryBase).isZero() &&
               "Primary base must be at offset 0!");
      }
    }
#endif        
}
