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
  if (CXXInfo) {
    CXXInfo->~CXXRecordLayoutInfo();
    Ctx.Deallocate(CXXInfo);
  }
  this->~ASTRecordLayout();
  Ctx.Deallocate(this);
}

ASTRecordLayout::ASTRecordLayout(const ASTContext &Ctx, CharUnits size,
                                 CharUnits alignment,
                                 CharUnits requiredAlignment,
                                 CharUnits datasize,
                                 ArrayRef<uint64_t> fieldoffsets)
    : Size(size), DataSize(datasize), Alignment(alignment),
      RequiredAlignment(requiredAlignment), CXXInfo(nullptr) {
  FieldOffsets.append(Ctx, fieldoffsets.begin(), fieldoffsets.end());
}

// Constructor for C++ records.
ASTRecordLayout::ASTRecordLayout(const ASTContext &Ctx,
                                 CharUnits size, CharUnits alignment,
                                 CharUnits requiredAlignment,
                                 bool hasOwnVFPtr, bool hasExtendableVFPtr,
                                 CharUnits vbptroffset,
                                 CharUnits datasize,
                                 ArrayRef<uint64_t> fieldoffsets,
                                 CharUnits nonvirtualsize,
                                 CharUnits nonvirtualalignment,
                                 CharUnits SizeOfLargestEmptySubobject,
                                 const CXXRecordDecl *PrimaryBase,
                                 bool IsPrimaryBaseVirtual,
                                 const CXXRecordDecl *BaseSharingVBPtr,
                                 bool EndsWithZeroSizedObject,
                                 bool LeadsWithZeroSizedBase,
                                 const BaseOffsetsMapTy& BaseOffsets,
                                 const VBaseOffsetsMapTy& VBaseOffsets)
  : Size(size), DataSize(datasize), Alignment(alignment),
    RequiredAlignment(requiredAlignment), CXXInfo(new (Ctx) CXXRecordLayoutInfo)
{
  FieldOffsets.append(Ctx, fieldoffsets.begin(), fieldoffsets.end());

  CXXInfo->PrimaryBase.setPointer(PrimaryBase);
  CXXInfo->PrimaryBase.setInt(IsPrimaryBaseVirtual);
  CXXInfo->NonVirtualSize = nonvirtualsize;
  CXXInfo->NonVirtualAlignment = nonvirtualalignment;
  CXXInfo->SizeOfLargestEmptySubobject = SizeOfLargestEmptySubobject;
  CXXInfo->BaseOffsets = BaseOffsets;
  CXXInfo->VBaseOffsets = VBaseOffsets;
  CXXInfo->HasOwnVFPtr = hasOwnVFPtr;
  CXXInfo->VBPtrOffset = vbptroffset;
  CXXInfo->HasExtendableVFPtr = hasExtendableVFPtr;
  CXXInfo->BaseSharingVBPtr = BaseSharingVBPtr;
  CXXInfo->EndsWithZeroSizedObject = EndsWithZeroSizedObject;
  CXXInfo->LeadsWithZeroSizedBase = LeadsWithZeroSizedBase;


#ifndef NDEBUG
    if (const CXXRecordDecl *PrimaryBase = getPrimaryBase()) {
      if (isPrimaryBaseVirtual()) {
        if (Ctx.getTargetInfo().getCXXABI().hasPrimaryVBases()) {
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
