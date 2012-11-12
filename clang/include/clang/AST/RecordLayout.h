//===--- RecordLayout.h - Layout information for a struct/union -*- C++ -*-===//
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

#ifndef LLVM_CLANG_AST_LAYOUTINFO_H
#define LLVM_CLANG_AST_LAYOUTINFO_H

#include "clang/AST/CharUnits.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
  class ASTContext;
  class FieldDecl;
  class RecordDecl;
  class CXXRecordDecl;

/// ASTRecordLayout -
/// This class contains layout information for one RecordDecl,
/// which is a struct/union/class.  The decl represented must be a definition,
/// not a forward declaration.
/// This class is also used to contain layout information for one
/// ObjCInterfaceDecl. FIXME - Find appropriate name.
/// These objects are managed by ASTContext.
class ASTRecordLayout {
public:
  struct VBaseInfo {
    /// The offset to this virtual base in the complete-object layout
    /// of this class.
    CharUnits VBaseOffset;

  private:
    /// Whether this virtual base requires a vtordisp field in the
    /// Microsoft ABI.  These fields are required for certain operations
    /// in constructors and destructors.
    bool HasVtorDisp;

  public:
    bool hasVtorDisp() const { return HasVtorDisp; }

    VBaseInfo() : HasVtorDisp(false) {}

    VBaseInfo(CharUnits VBaseOffset, bool hasVtorDisp) :
     VBaseOffset(VBaseOffset), HasVtorDisp(hasVtorDisp) {}
  };

  typedef llvm::DenseMap<const CXXRecordDecl *, VBaseInfo>
    VBaseOffsetsMapTy;

private:
  /// Size - Size of record in characters.
  CharUnits Size;

  /// DataSize - Size of record in characters without tail padding.
  CharUnits DataSize;

  // Alignment - Alignment of record in characters.
  CharUnits Alignment;

  /// FieldOffsets - Array of field offsets in bits.
  uint64_t *FieldOffsets;

  // FieldCount - Number of fields.
  unsigned FieldCount;

  /// CXXRecordLayoutInfo - Contains C++ specific layout information.
  struct CXXRecordLayoutInfo {
    /// NonVirtualSize - The non-virtual size (in chars) of an object, which is
    /// the size of the object without virtual bases.
    CharUnits NonVirtualSize;

    /// NonVirtualAlign - The non-virtual alignment (in chars) of an object,
    /// which is the alignment of the object without virtual bases.
    CharUnits NonVirtualAlign;

    /// SizeOfLargestEmptySubobject - The size of the largest empty subobject
    /// (either a base or a member). Will be zero if the class doesn't contain
    /// any empty subobjects.
    CharUnits SizeOfLargestEmptySubobject;

    /// VBPtrOffset - Virtual base table offset (Microsoft-only).
    CharUnits VBPtrOffset;

    /// HasOwnVFPtr - Does this class provide a virtual function table
    /// (vtable in Itanium, vftbl in Microsoft) that is independent from
    /// its base classes?
    bool HasOwnVFPtr; // TODO: stash this somewhere more efficient
    
    /// PrimaryBase - The primary base info for this record.
    llvm::PointerIntPair<const CXXRecordDecl *, 1, bool> PrimaryBase;
    
    /// FIXME: This should really use a SmallPtrMap, once we have one in LLVM :)
    typedef llvm::DenseMap<const CXXRecordDecl *, CharUnits> BaseOffsetsMapTy;
    
    /// BaseOffsets - Contains a map from base classes to their offset.
    BaseOffsetsMapTy BaseOffsets;

    /// VBaseOffsets - Contains a map from vbase classes to their offset.
    VBaseOffsetsMapTy VBaseOffsets;
  };

  /// CXXInfo - If the record layout is for a C++ record, this will have
  /// C++ specific information about the record.
  CXXRecordLayoutInfo *CXXInfo;

  friend class ASTContext;

  ASTRecordLayout(const ASTContext &Ctx, CharUnits size, CharUnits alignment,
                  CharUnits datasize, const uint64_t *fieldoffsets,
                  unsigned fieldcount);

  // Constructor for C++ records.
  typedef CXXRecordLayoutInfo::BaseOffsetsMapTy BaseOffsetsMapTy;
  ASTRecordLayout(const ASTContext &Ctx,
                  CharUnits size, CharUnits alignment,
                  bool hasOwnVFPtr, CharUnits vbptroffset,
                  CharUnits datasize,
                  const uint64_t *fieldoffsets, unsigned fieldcount,
                  CharUnits nonvirtualsize, CharUnits nonvirtualalign,
                  CharUnits SizeOfLargestEmptySubobject,
                  const CXXRecordDecl *PrimaryBase,
                  bool IsPrimaryBaseVirtual,
                  const BaseOffsetsMapTy& BaseOffsets,
                  const VBaseOffsetsMapTy& VBaseOffsets);

  ~ASTRecordLayout() {}

  void Destroy(ASTContext &Ctx);
  
  ASTRecordLayout(const ASTRecordLayout &) LLVM_DELETED_FUNCTION;
  void operator=(const ASTRecordLayout &) LLVM_DELETED_FUNCTION;
public:

  /// getAlignment - Get the record alignment in characters.
  CharUnits getAlignment() const { return Alignment; }

  /// getSize - Get the record size in characters.
  CharUnits getSize() const { return Size; }

  /// getFieldCount - Get the number of fields in the layout.
  unsigned getFieldCount() const { return FieldCount; }

  /// getFieldOffset - Get the offset of the given field index, in
  /// bits.
  uint64_t getFieldOffset(unsigned FieldNo) const {
    assert (FieldNo < FieldCount && "Invalid Field No");
    return FieldOffsets[FieldNo];
  }

  /// getDataSize() - Get the record data size, which is the record size
  /// without tail padding, in characters.
  CharUnits getDataSize() const {
    return DataSize;
  }

  /// getNonVirtualSize - Get the non-virtual size (in chars) of an object,
  /// which is the size of the object without virtual bases.
  CharUnits getNonVirtualSize() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");

    return CXXInfo->NonVirtualSize;
  }

  /// getNonVirtualSize - Get the non-virtual alignment (in chars) of an object,
  /// which is the alignment of the object without virtual bases.
  CharUnits getNonVirtualAlign() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");

    return CXXInfo->NonVirtualAlign;
  }

  /// getPrimaryBase - Get the primary base for this record.
  const CXXRecordDecl *getPrimaryBase() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");

    return CXXInfo->PrimaryBase.getPointer();
  }

  /// isPrimaryBaseVirtual - Get whether the primary base for this record
  /// is virtual or not.
  bool isPrimaryBaseVirtual() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");

    return CXXInfo->PrimaryBase.getInt();
  }

  /// getBaseClassOffset - Get the offset, in chars, for the given base class.
  CharUnits getBaseClassOffset(const CXXRecordDecl *Base) const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    assert(CXXInfo->BaseOffsets.count(Base) && "Did not find base!");

    return CXXInfo->BaseOffsets[Base];
  }

  /// getVBaseClassOffset - Get the offset, in chars, for the given base class.
  CharUnits getVBaseClassOffset(const CXXRecordDecl *VBase) const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    assert(CXXInfo->VBaseOffsets.count(VBase) && "Did not find base!");

    return CXXInfo->VBaseOffsets[VBase].VBaseOffset;
  }

  CharUnits getSizeOfLargestEmptySubobject() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    return CXXInfo->SizeOfLargestEmptySubobject;
  }

  /// hasOwnVFPtr - Does this class provide its own virtual-function
  /// table pointer, rather than inheriting one from a primary base
  /// class?  If so, it is at offset zero.
  ///
  /// This implies that the ABI has no primary base class, meaning
  /// that it has no base classes that are suitable under the conditions
  /// of the ABI.
  bool hasOwnVFPtr() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    return CXXInfo->HasOwnVFPtr;
  }

  /// getVBPtrOffset - Get the offset for virtual base table pointer.
  /// This is only meaningful with the Microsoft ABI.
  CharUnits getVBPtrOffset() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    return CXXInfo->VBPtrOffset;
  }

  const VBaseOffsetsMapTy &getVBaseOffsetsMap() const {
    assert(CXXInfo && "Record layout does not have C++ specific info!");
    return CXXInfo->VBaseOffsets;
  }
};

}  // end namespace clang

#endif
