//=== RecordLayoutBuilder.cpp - Helper class for building record layouts ---==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/Support/Format.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/CrashRecoveryContext.h"

using namespace clang;

namespace {

/// BaseSubobjectInfo - Represents a single base subobject in a complete class.
/// For a class hierarchy like
///
/// class A { };
/// class B : A { };
/// class C : A, B { };
///
/// The BaseSubobjectInfo graph for C will have three BaseSubobjectInfo
/// instances, one for B and two for A.
///
/// If a base is virtual, it will only have one BaseSubobjectInfo allocated.
struct BaseSubobjectInfo {
  /// Class - The class for this base info.
  const CXXRecordDecl *Class;

  /// IsVirtual - Whether the BaseInfo represents a virtual base or not.
  bool IsVirtual;

  /// Bases - Information about the base subobjects.
  SmallVector<BaseSubobjectInfo*, 4> Bases;

  /// PrimaryVirtualBaseInfo - Holds the base info for the primary virtual base
  /// of this base info (if one exists).
  BaseSubobjectInfo *PrimaryVirtualBaseInfo;

  // FIXME: Document.
  const BaseSubobjectInfo *Derived;
};

/// EmptySubobjectMap - Keeps track of which empty subobjects exist at different
/// offsets while laying out a C++ class.
class EmptySubobjectMap {
  const ASTContext &Context;
  uint64_t CharWidth;
  
  /// Class - The class whose empty entries we're keeping track of.
  const CXXRecordDecl *Class;

  /// EmptyClassOffsets - A map from offsets to empty record decls.
  typedef SmallVector<const CXXRecordDecl *, 1> ClassVectorTy;
  typedef llvm::DenseMap<CharUnits, ClassVectorTy> EmptyClassOffsetsMapTy;
  EmptyClassOffsetsMapTy EmptyClassOffsets;
  
  /// MaxEmptyClassOffset - The highest offset known to contain an empty
  /// base subobject.
  CharUnits MaxEmptyClassOffset;
  
  /// ComputeEmptySubobjectSizes - Compute the size of the largest base or
  /// member subobject that is empty.
  void ComputeEmptySubobjectSizes();
  
  void AddSubobjectAtOffset(const CXXRecordDecl *RD, CharUnits Offset);
  
  void UpdateEmptyBaseSubobjects(const BaseSubobjectInfo *Info,
                                 CharUnits Offset, bool PlacingEmptyBase);
  
  void UpdateEmptyFieldSubobjects(const CXXRecordDecl *RD, 
                                  const CXXRecordDecl *Class,
                                  CharUnits Offset);
  void UpdateEmptyFieldSubobjects(const FieldDecl *FD, CharUnits Offset);
  
  /// AnyEmptySubobjectsBeyondOffset - Returns whether there are any empty
  /// subobjects beyond the given offset.
  bool AnyEmptySubobjectsBeyondOffset(CharUnits Offset) const {
    return Offset <= MaxEmptyClassOffset;
  }

  CharUnits 
  getFieldOffset(const ASTRecordLayout &Layout, unsigned FieldNo) const {
    uint64_t FieldOffset = Layout.getFieldOffset(FieldNo);
    assert(FieldOffset % CharWidth == 0 && 
           "Field offset not at char boundary!");

    return Context.toCharUnitsFromBits(FieldOffset);
  }

protected:
  bool CanPlaceSubobjectAtOffset(const CXXRecordDecl *RD,
                                 CharUnits Offset) const;

  bool CanPlaceBaseSubobjectAtOffset(const BaseSubobjectInfo *Info,
                                     CharUnits Offset);

  bool CanPlaceFieldSubobjectAtOffset(const CXXRecordDecl *RD, 
                                      const CXXRecordDecl *Class,
                                      CharUnits Offset) const;
  bool CanPlaceFieldSubobjectAtOffset(const FieldDecl *FD,
                                      CharUnits Offset) const;

public:
  /// This holds the size of the largest empty subobject (either a base
  /// or a member). Will be zero if the record being built doesn't contain
  /// any empty classes.
  CharUnits SizeOfLargestEmptySubobject;

  EmptySubobjectMap(const ASTContext &Context, const CXXRecordDecl *Class)
  : Context(Context), CharWidth(Context.getCharWidth()), Class(Class) {
      ComputeEmptySubobjectSizes();
  }

  /// CanPlaceBaseAtOffset - Return whether the given base class can be placed
  /// at the given offset.
  /// Returns false if placing the record will result in two components
  /// (direct or indirect) of the same type having the same offset.
  bool CanPlaceBaseAtOffset(const BaseSubobjectInfo *Info,
                            CharUnits Offset);

  /// CanPlaceFieldAtOffset - Return whether a field can be placed at the given
  /// offset.
  bool CanPlaceFieldAtOffset(const FieldDecl *FD, CharUnits Offset);
};

void EmptySubobjectMap::ComputeEmptySubobjectSizes() {
  // Check the bases.
  for (CXXRecordDecl::base_class_const_iterator I = Class->bases_begin(),
       E = Class->bases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    CharUnits EmptySize;
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(BaseDecl);
    if (BaseDecl->isEmpty()) {
      // If the class decl is empty, get its size.
      EmptySize = Layout.getSize();
    } else {
      // Otherwise, we get the largest empty subobject for the decl.
      EmptySize = Layout.getSizeOfLargestEmptySubobject();
    }

    if (EmptySize > SizeOfLargestEmptySubobject)
      SizeOfLargestEmptySubobject = EmptySize;
  }

  // Check the fields.
  for (CXXRecordDecl::field_iterator I = Class->field_begin(),
       E = Class->field_end(); I != E; ++I) {

    const RecordType *RT =
      Context.getBaseElementType(I->getType())->getAs<RecordType>();

    // We only care about record types.
    if (!RT)
      continue;

    CharUnits EmptySize;
    const CXXRecordDecl *MemberDecl = cast<CXXRecordDecl>(RT->getDecl());
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(MemberDecl);
    if (MemberDecl->isEmpty()) {
      // If the class decl is empty, get its size.
      EmptySize = Layout.getSize();
    } else {
      // Otherwise, we get the largest empty subobject for the decl.
      EmptySize = Layout.getSizeOfLargestEmptySubobject();
    }

    if (EmptySize > SizeOfLargestEmptySubobject)
      SizeOfLargestEmptySubobject = EmptySize;
  }
}

bool
EmptySubobjectMap::CanPlaceSubobjectAtOffset(const CXXRecordDecl *RD, 
                                             CharUnits Offset) const {
  // We only need to check empty bases.
  if (!RD->isEmpty())
    return true;

  EmptyClassOffsetsMapTy::const_iterator I = EmptyClassOffsets.find(Offset);
  if (I == EmptyClassOffsets.end())
    return true;
  
  const ClassVectorTy& Classes = I->second;
  if (std::find(Classes.begin(), Classes.end(), RD) == Classes.end())
    return true;

  // There is already an empty class of the same type at this offset.
  return false;
}
  
void EmptySubobjectMap::AddSubobjectAtOffset(const CXXRecordDecl *RD, 
                                             CharUnits Offset) {
  // We only care about empty bases.
  if (!RD->isEmpty())
    return;

  // If we have empty structures inside an union, we can assign both
  // the same offset. Just avoid pushing them twice in the list.
  ClassVectorTy& Classes = EmptyClassOffsets[Offset];
  if (std::find(Classes.begin(), Classes.end(), RD) != Classes.end())
    return;
  
  Classes.push_back(RD);
  
  // Update the empty class offset.
  if (Offset > MaxEmptyClassOffset)
    MaxEmptyClassOffset = Offset;
}

bool
EmptySubobjectMap::CanPlaceBaseSubobjectAtOffset(const BaseSubobjectInfo *Info,
                                                 CharUnits Offset) {
  // We don't have to keep looking past the maximum offset that's known to
  // contain an empty class.
  if (!AnyEmptySubobjectsBeyondOffset(Offset))
    return true;

  if (!CanPlaceSubobjectAtOffset(Info->Class, Offset))
    return false;

  // Traverse all non-virtual bases.
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(Info->Class);
  for (unsigned I = 0, E = Info->Bases.size(); I != E; ++I) {
    BaseSubobjectInfo* Base = Info->Bases[I];
    if (Base->IsVirtual)
      continue;

    CharUnits BaseOffset = Offset + Layout.getBaseClassOffset(Base->Class);

    if (!CanPlaceBaseSubobjectAtOffset(Base, BaseOffset))
      return false;
  }

  if (Info->PrimaryVirtualBaseInfo) {
    BaseSubobjectInfo *PrimaryVirtualBaseInfo = Info->PrimaryVirtualBaseInfo;

    if (Info == PrimaryVirtualBaseInfo->Derived) {
      if (!CanPlaceBaseSubobjectAtOffset(PrimaryVirtualBaseInfo, Offset))
        return false;
    }
  }
  
  // Traverse all member variables.
  unsigned FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = Info->Class->field_begin(), 
       E = Info->Class->field_end(); I != E; ++I, ++FieldNo) {
    if (I->isBitField())
      continue;
  
    CharUnits FieldOffset = Offset + getFieldOffset(Layout, FieldNo);
    if (!CanPlaceFieldSubobjectAtOffset(*I, FieldOffset))
      return false;
  }
  
  return true;
}

void EmptySubobjectMap::UpdateEmptyBaseSubobjects(const BaseSubobjectInfo *Info, 
                                                  CharUnits Offset,
                                                  bool PlacingEmptyBase) {
  if (!PlacingEmptyBase && Offset >= SizeOfLargestEmptySubobject) {
    // We know that the only empty subobjects that can conflict with empty
    // subobject of non-empty bases, are empty bases that can be placed at
    // offset zero. Because of this, we only need to keep track of empty base 
    // subobjects with offsets less than the size of the largest empty
    // subobject for our class.    
    return;
  }

  AddSubobjectAtOffset(Info->Class, Offset);

  // Traverse all non-virtual bases.
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(Info->Class);
  for (unsigned I = 0, E = Info->Bases.size(); I != E; ++I) {
    BaseSubobjectInfo* Base = Info->Bases[I];
    if (Base->IsVirtual)
      continue;

    CharUnits BaseOffset = Offset + Layout.getBaseClassOffset(Base->Class);
    UpdateEmptyBaseSubobjects(Base, BaseOffset, PlacingEmptyBase);
  }

  if (Info->PrimaryVirtualBaseInfo) {
    BaseSubobjectInfo *PrimaryVirtualBaseInfo = Info->PrimaryVirtualBaseInfo;
    
    if (Info == PrimaryVirtualBaseInfo->Derived)
      UpdateEmptyBaseSubobjects(PrimaryVirtualBaseInfo, Offset,
                                PlacingEmptyBase);
  }

  // Traverse all member variables.
  unsigned FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = Info->Class->field_begin(), 
       E = Info->Class->field_end(); I != E; ++I, ++FieldNo) {
    if (I->isBitField())
      continue;

    CharUnits FieldOffset = Offset + getFieldOffset(Layout, FieldNo);
    UpdateEmptyFieldSubobjects(*I, FieldOffset);
  }
}

bool EmptySubobjectMap::CanPlaceBaseAtOffset(const BaseSubobjectInfo *Info,
                                             CharUnits Offset) {
  // If we know this class doesn't have any empty subobjects we don't need to
  // bother checking.
  if (SizeOfLargestEmptySubobject.isZero())
    return true;

  if (!CanPlaceBaseSubobjectAtOffset(Info, Offset))
    return false;

  // We are able to place the base at this offset. Make sure to update the
  // empty base subobject map.
  UpdateEmptyBaseSubobjects(Info, Offset, Info->Class->isEmpty());
  return true;
}

bool
EmptySubobjectMap::CanPlaceFieldSubobjectAtOffset(const CXXRecordDecl *RD, 
                                                  const CXXRecordDecl *Class,
                                                  CharUnits Offset) const {
  // We don't have to keep looking past the maximum offset that's known to
  // contain an empty class.
  if (!AnyEmptySubobjectsBeyondOffset(Offset))
    return true;

  if (!CanPlaceSubobjectAtOffset(RD, Offset))
    return false;
  
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  // Traverse all non-virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    CharUnits BaseOffset = Offset + Layout.getBaseClassOffset(BaseDecl);
    if (!CanPlaceFieldSubobjectAtOffset(BaseDecl, Class, BaseOffset))
      return false;
  }

  if (RD == Class) {
    // This is the most derived class, traverse virtual bases as well.
    for (CXXRecordDecl::base_class_const_iterator I = RD->vbases_begin(),
         E = RD->vbases_end(); I != E; ++I) {
      const CXXRecordDecl *VBaseDecl =
        cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
      
      CharUnits VBaseOffset = Offset + Layout.getVBaseClassOffset(VBaseDecl);
      if (!CanPlaceFieldSubobjectAtOffset(VBaseDecl, Class, VBaseOffset))
        return false;
    }
  }
    
  // Traverse all member variables.
  unsigned FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();
       I != E; ++I, ++FieldNo) {
    if (I->isBitField())
      continue;

    CharUnits FieldOffset = Offset + getFieldOffset(Layout, FieldNo);
    
    if (!CanPlaceFieldSubobjectAtOffset(*I, FieldOffset))
      return false;
  }

  return true;
}

bool
EmptySubobjectMap::CanPlaceFieldSubobjectAtOffset(const FieldDecl *FD,
                                                  CharUnits Offset) const {
  // We don't have to keep looking past the maximum offset that's known to
  // contain an empty class.
  if (!AnyEmptySubobjectsBeyondOffset(Offset))
    return true;
  
  QualType T = FD->getType();
  if (const RecordType *RT = T->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    return CanPlaceFieldSubobjectAtOffset(RD, RD, Offset);
  }

  // If we have an array type we need to look at every element.
  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(T)) {
    QualType ElemTy = Context.getBaseElementType(AT);
    const RecordType *RT = ElemTy->getAs<RecordType>();
    if (!RT)
      return true;
  
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

    uint64_t NumElements = Context.getConstantArrayElementCount(AT);
    CharUnits ElementOffset = Offset;
    for (uint64_t I = 0; I != NumElements; ++I) {
      // We don't have to keep looking past the maximum offset that's known to
      // contain an empty class.
      if (!AnyEmptySubobjectsBeyondOffset(ElementOffset))
        return true;
      
      if (!CanPlaceFieldSubobjectAtOffset(RD, RD, ElementOffset))
        return false;

      ElementOffset += Layout.getSize();
    }
  }

  return true;
}

bool
EmptySubobjectMap::CanPlaceFieldAtOffset(const FieldDecl *FD, 
                                         CharUnits Offset) {
  if (!CanPlaceFieldSubobjectAtOffset(FD, Offset))
    return false;
  
  // We are able to place the member variable at this offset.
  // Make sure to update the empty base subobject map.
  UpdateEmptyFieldSubobjects(FD, Offset);
  return true;
}

void EmptySubobjectMap::UpdateEmptyFieldSubobjects(const CXXRecordDecl *RD, 
                                                   const CXXRecordDecl *Class,
                                                   CharUnits Offset) {
  // We know that the only empty subobjects that can conflict with empty
  // field subobjects are subobjects of empty bases that can be placed at offset
  // zero. Because of this, we only need to keep track of empty field 
  // subobjects with offsets less than the size of the largest empty
  // subobject for our class.
  if (Offset >= SizeOfLargestEmptySubobject)
    return;

  AddSubobjectAtOffset(RD, Offset);

  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

  // Traverse all non-virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    CharUnits BaseOffset = Offset + Layout.getBaseClassOffset(BaseDecl);
    UpdateEmptyFieldSubobjects(BaseDecl, Class, BaseOffset);
  }

  if (RD == Class) {
    // This is the most derived class, traverse virtual bases as well.
    for (CXXRecordDecl::base_class_const_iterator I = RD->vbases_begin(),
         E = RD->vbases_end(); I != E; ++I) {
      const CXXRecordDecl *VBaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
      
      CharUnits VBaseOffset = Offset + Layout.getVBaseClassOffset(VBaseDecl);
      UpdateEmptyFieldSubobjects(VBaseDecl, Class, VBaseOffset);
    }
  }
  
  // Traverse all member variables.
  unsigned FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = RD->field_begin(), E = RD->field_end();
       I != E; ++I, ++FieldNo) {
    if (I->isBitField())
      continue;

    CharUnits FieldOffset = Offset + getFieldOffset(Layout, FieldNo);

    UpdateEmptyFieldSubobjects(*I, FieldOffset);
  }
}
  
void EmptySubobjectMap::UpdateEmptyFieldSubobjects(const FieldDecl *FD,
                                                   CharUnits Offset) {
  QualType T = FD->getType();
  if (const RecordType *RT = T->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    UpdateEmptyFieldSubobjects(RD, RD, Offset);
    return;
  }

  // If we have an array type we need to update every element.
  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(T)) {
    QualType ElemTy = Context.getBaseElementType(AT);
    const RecordType *RT = ElemTy->getAs<RecordType>();
    if (!RT)
      return;
    
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    
    uint64_t NumElements = Context.getConstantArrayElementCount(AT);
    CharUnits ElementOffset = Offset;
    
    for (uint64_t I = 0; I != NumElements; ++I) {
      // We know that the only empty subobjects that can conflict with empty
      // field subobjects are subobjects of empty bases that can be placed at 
      // offset zero. Because of this, we only need to keep track of empty field
      // subobjects with offsets less than the size of the largest empty
      // subobject for our class.
      if (ElementOffset >= SizeOfLargestEmptySubobject)
        return;

      UpdateEmptyFieldSubobjects(RD, RD, ElementOffset);
      ElementOffset += Layout.getSize();
    }
  }
}

typedef llvm::SmallPtrSet<const CXXRecordDecl*, 4> ClassSetTy;

class RecordLayoutBuilder {
protected:
  // FIXME: Remove this and make the appropriate fields public.
  friend class clang::ASTContext;

  const ASTContext &Context;

  EmptySubobjectMap *EmptySubobjects;

  /// Size - The current size of the record layout.
  uint64_t Size;

  /// Alignment - The current alignment of the record layout.
  CharUnits Alignment;

  /// \brief The alignment if attribute packed is not used.
  CharUnits UnpackedAlignment;

  SmallVector<uint64_t, 16> FieldOffsets;

  /// \brief Whether the external AST source has provided a layout for this
  /// record.
  unsigned ExternalLayout : 1;

  /// \brief Whether we need to infer alignment, even when we have an 
  /// externally-provided layout.
  unsigned InferAlignment : 1;
  
  /// Packed - Whether the record is packed or not.
  unsigned Packed : 1;

  unsigned IsUnion : 1;

  unsigned IsMac68kAlign : 1;
  
  unsigned IsMsStruct : 1;

  /// UnfilledBitsInLastByte - If the last field laid out was a bitfield,
  /// this contains the number of bits in the last byte that can be used for
  /// an adjacent bitfield if necessary.
  unsigned char UnfilledBitsInLastByte;

  /// MaxFieldAlignment - The maximum allowed field alignment. This is set by
  /// #pragma pack.
  CharUnits MaxFieldAlignment;

  /// DataSize - The data size of the record being laid out.
  uint64_t DataSize;

  CharUnits NonVirtualSize;
  CharUnits NonVirtualAlignment;

  FieldDecl *ZeroLengthBitfield;

  /// PrimaryBase - the primary base class (if one exists) of the class
  /// we're laying out.
  const CXXRecordDecl *PrimaryBase;

  /// PrimaryBaseIsVirtual - Whether the primary base of the class we're laying
  /// out is virtual.
  bool PrimaryBaseIsVirtual;

  /// HasOwnVFPtr - Whether the class provides its own vtable/vftbl
  /// pointer, as opposed to inheriting one from a primary base class.
  bool HasOwnVFPtr;

  /// VBPtrOffset - Virtual base table offset. Only for MS layout.
  CharUnits VBPtrOffset;

  typedef llvm::DenseMap<const CXXRecordDecl *, CharUnits> BaseOffsetsMapTy;

  /// Bases - base classes and their offsets in the record.
  BaseOffsetsMapTy Bases;

  // VBases - virtual base classes and their offsets in the record.
  ASTRecordLayout::VBaseOffsetsMapTy VBases;

  /// IndirectPrimaryBases - Virtual base classes, direct or indirect, that are
  /// primary base classes for some other direct or indirect base class.
  CXXIndirectPrimaryBaseSet IndirectPrimaryBases;

  /// FirstNearlyEmptyVBase - The first nearly empty virtual base class in
  /// inheritance graph order. Used for determining the primary base class.
  const CXXRecordDecl *FirstNearlyEmptyVBase;

  /// VisitedVirtualBases - A set of all the visited virtual bases, used to
  /// avoid visiting virtual bases more than once.
  llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBases;

  /// \brief Externally-provided size.
  uint64_t ExternalSize;
  
  /// \brief Externally-provided alignment.
  uint64_t ExternalAlign;
  
  /// \brief Externally-provided field offsets.
  llvm::DenseMap<const FieldDecl *, uint64_t> ExternalFieldOffsets;

  /// \brief Externally-provided direct, non-virtual base offsets.
  llvm::DenseMap<const CXXRecordDecl *, CharUnits> ExternalBaseOffsets;

  /// \brief Externally-provided virtual base offsets.
  llvm::DenseMap<const CXXRecordDecl *, CharUnits> ExternalVirtualBaseOffsets;

  RecordLayoutBuilder(const ASTContext &Context,
                      EmptySubobjectMap *EmptySubobjects)
    : Context(Context), EmptySubobjects(EmptySubobjects), Size(0), 
      Alignment(CharUnits::One()), UnpackedAlignment(CharUnits::One()),
      ExternalLayout(false), InferAlignment(false), 
      Packed(false), IsUnion(false), IsMac68kAlign(false), IsMsStruct(false),
      UnfilledBitsInLastByte(0), MaxFieldAlignment(CharUnits::Zero()), 
      DataSize(0), NonVirtualSize(CharUnits::Zero()), 
      NonVirtualAlignment(CharUnits::One()), 
      ZeroLengthBitfield(0), PrimaryBase(0), 
      PrimaryBaseIsVirtual(false),
      HasOwnVFPtr(false),
      VBPtrOffset(CharUnits::fromQuantity(-1)),
      FirstNearlyEmptyVBase(0) { }

  /// Reset this RecordLayoutBuilder to a fresh state, using the given
  /// alignment as the initial alignment.  This is used for the
  /// correct layout of vb-table pointers in MSVC.
  void resetWithTargetAlignment(CharUnits TargetAlignment) {
    const ASTContext &Context = this->Context;
    EmptySubobjectMap *EmptySubobjects = this->EmptySubobjects;
    this->~RecordLayoutBuilder();
    new (this) RecordLayoutBuilder(Context, EmptySubobjects);
    Alignment = UnpackedAlignment = TargetAlignment;
  }

  void Layout(const RecordDecl *D);
  void Layout(const CXXRecordDecl *D);
  void Layout(const ObjCInterfaceDecl *D);

  void LayoutFields(const RecordDecl *D);
  void LayoutField(const FieldDecl *D);
  void LayoutWideBitField(uint64_t FieldSize, uint64_t TypeSize,
                          bool FieldPacked, const FieldDecl *D);
  void LayoutBitField(const FieldDecl *D);

  bool isMicrosoftCXXABI() const {
    return Context.getTargetInfo().getCXXABI() == CXXABI_Microsoft;
  }

  void MSLayoutVirtualBases(const CXXRecordDecl *RD);

  /// BaseSubobjectInfoAllocator - Allocator for BaseSubobjectInfo objects.
  llvm::SpecificBumpPtrAllocator<BaseSubobjectInfo> BaseSubobjectInfoAllocator;
  
  typedef llvm::DenseMap<const CXXRecordDecl *, BaseSubobjectInfo *>
    BaseSubobjectInfoMapTy;

  /// VirtualBaseInfo - Map from all the (direct or indirect) virtual bases
  /// of the class we're laying out to their base subobject info.
  BaseSubobjectInfoMapTy VirtualBaseInfo;
  
  /// NonVirtualBaseInfo - Map from all the direct non-virtual bases of the
  /// class we're laying out to their base subobject info.
  BaseSubobjectInfoMapTy NonVirtualBaseInfo;

  /// ComputeBaseSubobjectInfo - Compute the base subobject information for the
  /// bases of the given class.
  void ComputeBaseSubobjectInfo(const CXXRecordDecl *RD);

  /// ComputeBaseSubobjectInfo - Compute the base subobject information for a
  /// single class and all of its base classes.
  BaseSubobjectInfo *ComputeBaseSubobjectInfo(const CXXRecordDecl *RD, 
                                              bool IsVirtual,
                                              BaseSubobjectInfo *Derived);

  /// DeterminePrimaryBase - Determine the primary base of the given class.
  void DeterminePrimaryBase(const CXXRecordDecl *RD);

  void SelectPrimaryVBase(const CXXRecordDecl *RD);

  void EnsureVTablePointerAlignment(CharUnits UnpackedBaseAlign);

  /// LayoutNonVirtualBases - Determines the primary base class (if any) and
  /// lays it out. Will then proceed to lay out all non-virtual base clasess.
  void LayoutNonVirtualBases(const CXXRecordDecl *RD);

  /// LayoutNonVirtualBase - Lays out a single non-virtual base.
  void LayoutNonVirtualBase(const BaseSubobjectInfo *Base);

  void AddPrimaryVirtualBaseOffsets(const BaseSubobjectInfo *Info,
                                    CharUnits Offset);

  bool needsVFTable(const CXXRecordDecl *RD) const;
  bool hasNewVirtualFunction(const CXXRecordDecl *RD,
                             bool IgnoreDestructor = false) const;
  bool isPossiblePrimaryBase(const CXXRecordDecl *Base) const;

  void computeVtordisps(const CXXRecordDecl *RD, 
                        ClassSetTy &VtordispVBases);

  /// LayoutVirtualBases - Lays out all the virtual bases.
  void LayoutVirtualBases(const CXXRecordDecl *RD,
                          const CXXRecordDecl *MostDerivedClass);

  /// LayoutVirtualBase - Lays out a single virtual base.
  void LayoutVirtualBase(const BaseSubobjectInfo *Base, 
                         bool IsVtordispNeed = false);

  /// LayoutBase - Will lay out a base and return the offset where it was
  /// placed, in chars.
  CharUnits LayoutBase(const BaseSubobjectInfo *Base);

  /// InitializeLayout - Initialize record layout for the given record decl.
  void InitializeLayout(const Decl *D);

  /// FinishLayout - Finalize record layout. Adjust record size based on the
  /// alignment.
  void FinishLayout(const NamedDecl *D);

  void UpdateAlignment(CharUnits NewAlignment, CharUnits UnpackedNewAlignment);
  void UpdateAlignment(CharUnits NewAlignment) {
    UpdateAlignment(NewAlignment, NewAlignment);
  }

  /// \brief Retrieve the externally-supplied field offset for the given
  /// field.
  ///
  /// \param Field The field whose offset is being queried.
  /// \param ComputedOffset The offset that we've computed for this field.
  uint64_t updateExternalFieldOffset(const FieldDecl *Field, 
                                     uint64_t ComputedOffset);
  
  void CheckFieldPadding(uint64_t Offset, uint64_t UnpaddedOffset,
                          uint64_t UnpackedOffset, unsigned UnpackedAlign,
                          bool isPacked, const FieldDecl *D);

  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID);

  CharUnits getSize() const { 
    assert(Size % Context.getCharWidth() == 0);
    return Context.toCharUnitsFromBits(Size); 
  }
  uint64_t getSizeInBits() const { return Size; }

  void setSize(CharUnits NewSize) { Size = Context.toBits(NewSize); }
  void setSize(uint64_t NewSize) { Size = NewSize; }

  CharUnits getAligment() const { return Alignment; }

  CharUnits getDataSize() const { 
    assert(DataSize % Context.getCharWidth() == 0);
    return Context.toCharUnitsFromBits(DataSize); 
  }
  uint64_t getDataSizeInBits() const { return DataSize; }

  void setDataSize(CharUnits NewSize) { DataSize = Context.toBits(NewSize); }
  void setDataSize(uint64_t NewSize) { DataSize = NewSize; }

  RecordLayoutBuilder(const RecordLayoutBuilder &) LLVM_DELETED_FUNCTION;
  void operator=(const RecordLayoutBuilder &) LLVM_DELETED_FUNCTION;
public:
  static const CXXMethodDecl *ComputeKeyFunction(const CXXRecordDecl *RD);
};
} // end anonymous namespace

void
RecordLayoutBuilder::SelectPrimaryVBase(const CXXRecordDecl *RD) {
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {
    assert(!I->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");

    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    // Check if this is a nearly empty virtual base.
    if (I->isVirtual() && Context.isNearlyEmpty(Base)) {
      // If it's not an indirect primary base, then we've found our primary
      // base.
      if (!IndirectPrimaryBases.count(Base)) {
        PrimaryBase = Base;
        PrimaryBaseIsVirtual = true;
        return;
      }

      // Is this the first nearly empty virtual base?
      if (!FirstNearlyEmptyVBase)
        FirstNearlyEmptyVBase = Base;
    }

    SelectPrimaryVBase(Base);
    if (PrimaryBase)
      return;
  }
}

/// DeterminePrimaryBase - Determine the primary base of the given class.
void RecordLayoutBuilder::DeterminePrimaryBase(const CXXRecordDecl *RD) {
  // If the class isn't dynamic, it won't have a primary base.
  if (!RD->isDynamicClass())
    return;

  // Compute all the primary virtual bases for all of our direct and
  // indirect bases, and record all their primary virtual base classes.
  RD->getIndirectPrimaryBases(IndirectPrimaryBases);

  // If the record has a dynamic base class, attempt to choose a primary base
  // class. It is the first (in direct base class order) non-virtual dynamic
  // base class, if one exists.
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
         e = RD->bases_end(); i != e; ++i) {
    // Ignore virtual bases.
    if (i->isVirtual())
      continue;

    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());

    if (isPossiblePrimaryBase(Base)) {
      // We found it.
      PrimaryBase = Base;
      PrimaryBaseIsVirtual = false;
      return;
    }
  }

  // The Microsoft ABI doesn't have primary virtual bases.
  if (isMicrosoftCXXABI()) {
    assert(!PrimaryBase && "Should not get here with a primary base!");
    return;
  }

  // Under the Itanium ABI, if there is no non-virtual primary base class,
  // try to compute the primary virtual base.  The primary virtual base is
  // the first nearly empty virtual base that is not an indirect primary
  // virtual base class, if one exists.
  if (RD->getNumVBases() != 0) {
    SelectPrimaryVBase(RD);
    if (PrimaryBase)
      return;
  }

  // Otherwise, it is the first indirect primary base class, if one exists.
  if (FirstNearlyEmptyVBase) {
    PrimaryBase = FirstNearlyEmptyVBase;
    PrimaryBaseIsVirtual = true;
    return;
  }

  assert(!PrimaryBase && "Should not get here with a primary base!");
}

BaseSubobjectInfo *
RecordLayoutBuilder::ComputeBaseSubobjectInfo(const CXXRecordDecl *RD, 
                                              bool IsVirtual,
                                              BaseSubobjectInfo *Derived) {
  BaseSubobjectInfo *Info;
  
  if (IsVirtual) {
    // Check if we already have info about this virtual base.
    BaseSubobjectInfo *&InfoSlot = VirtualBaseInfo[RD];
    if (InfoSlot) {
      assert(InfoSlot->Class == RD && "Wrong class for virtual base info!");
      return InfoSlot;
    }

    // We don't, create it.
    InfoSlot = new (BaseSubobjectInfoAllocator.Allocate()) BaseSubobjectInfo;
    Info = InfoSlot;
  } else {
    Info = new (BaseSubobjectInfoAllocator.Allocate()) BaseSubobjectInfo;
  }
  
  Info->Class = RD;
  Info->IsVirtual = IsVirtual;
  Info->Derived = 0;
  Info->PrimaryVirtualBaseInfo = 0;
  
  const CXXRecordDecl *PrimaryVirtualBase = 0;
  BaseSubobjectInfo *PrimaryVirtualBaseInfo = 0;

  // Check if this base has a primary virtual base.
  if (RD->getNumVBases()) {
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    if (Layout.isPrimaryBaseVirtual()) {
      // This base does have a primary virtual base.
      PrimaryVirtualBase = Layout.getPrimaryBase();
      assert(PrimaryVirtualBase && "Didn't have a primary virtual base!");
      
      // Now check if we have base subobject info about this primary base.
      PrimaryVirtualBaseInfo = VirtualBaseInfo.lookup(PrimaryVirtualBase);
      
      if (PrimaryVirtualBaseInfo) {
        if (PrimaryVirtualBaseInfo->Derived) {
          // We did have info about this primary base, and it turns out that it
          // has already been claimed as a primary virtual base for another
          // base. 
          PrimaryVirtualBase = 0;        
        } else {
          // We can claim this base as our primary base.
          Info->PrimaryVirtualBaseInfo = PrimaryVirtualBaseInfo;
          PrimaryVirtualBaseInfo->Derived = Info;
        }
      }
    }
  }

  // Now go through all direct bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    bool IsVirtual = I->isVirtual();
    
    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    
    Info->Bases.push_back(ComputeBaseSubobjectInfo(BaseDecl, IsVirtual, Info));
  }
  
  if (PrimaryVirtualBase && !PrimaryVirtualBaseInfo) {
    // Traversing the bases must have created the base info for our primary
    // virtual base.
    PrimaryVirtualBaseInfo = VirtualBaseInfo.lookup(PrimaryVirtualBase);
    assert(PrimaryVirtualBaseInfo &&
           "Did not create a primary virtual base!");
      
    // Claim the primary virtual base as our primary virtual base.
    Info->PrimaryVirtualBaseInfo = PrimaryVirtualBaseInfo;
    PrimaryVirtualBaseInfo->Derived = Info;
  }
  
  return Info;
}

void RecordLayoutBuilder::ComputeBaseSubobjectInfo(const CXXRecordDecl *RD) {
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    bool IsVirtual = I->isVirtual();

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    
    // Compute the base subobject info for this base.
    BaseSubobjectInfo *Info = ComputeBaseSubobjectInfo(BaseDecl, IsVirtual, 0);

    if (IsVirtual) {
      // ComputeBaseInfo has already added this base for us.
      assert(VirtualBaseInfo.count(BaseDecl) &&
             "Did not add virtual base!");
    } else {
      // Add the base info to the map of non-virtual bases.
      assert(!NonVirtualBaseInfo.count(BaseDecl) &&
             "Non-virtual base already exists!");
      NonVirtualBaseInfo.insert(std::make_pair(BaseDecl, Info));
    }
  }
}

void
RecordLayoutBuilder::EnsureVTablePointerAlignment(CharUnits UnpackedBaseAlign) {
  CharUnits BaseAlign = (Packed) ? CharUnits::One() : UnpackedBaseAlign;

  // The maximum field alignment overrides base align.
  if (!MaxFieldAlignment.isZero()) {
    BaseAlign = std::min(BaseAlign, MaxFieldAlignment);
    UnpackedBaseAlign = std::min(UnpackedBaseAlign, MaxFieldAlignment);
  }

  // Round up the current record size to pointer alignment.
  setSize(getSize().RoundUpToAlignment(BaseAlign));
  setDataSize(getSize());

  // Update the alignment.
  UpdateAlignment(BaseAlign, UnpackedBaseAlign);
}

void
RecordLayoutBuilder::LayoutNonVirtualBases(const CXXRecordDecl *RD) {
  // Then, determine the primary base class.
  DeterminePrimaryBase(RD);

  // Compute base subobject info.
  ComputeBaseSubobjectInfo(RD);
  
  // If we have a primary base class, lay it out.
  if (PrimaryBase) {
    if (PrimaryBaseIsVirtual) {
      // If the primary virtual base was a primary virtual base of some other
      // base class we'll have to steal it.
      BaseSubobjectInfo *PrimaryBaseInfo = VirtualBaseInfo.lookup(PrimaryBase);
      PrimaryBaseInfo->Derived = 0;
      
      // We have a virtual primary base, insert it as an indirect primary base.
      IndirectPrimaryBases.insert(PrimaryBase);

      assert(!VisitedVirtualBases.count(PrimaryBase) &&
             "vbase already visited!");
      VisitedVirtualBases.insert(PrimaryBase);

      LayoutVirtualBase(PrimaryBaseInfo);
    } else {
      BaseSubobjectInfo *PrimaryBaseInfo = 
        NonVirtualBaseInfo.lookup(PrimaryBase);
      assert(PrimaryBaseInfo && 
             "Did not find base info for non-virtual primary base!");

      LayoutNonVirtualBase(PrimaryBaseInfo);
    }

  // If this class needs a vtable/vf-table and didn't get one from a
  // primary base, add it in now.
  } else if (needsVFTable(RD)) {
    assert(DataSize == 0 && "Vtable pointer must be at offset zero!");
    CharUnits PtrWidth = 
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerWidth(0));
    CharUnits PtrAlign = 
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerAlign(0));
    EnsureVTablePointerAlignment(PtrAlign);
    HasOwnVFPtr = true;
    setSize(getSize() + PtrWidth);
    setDataSize(getSize());
  }

  bool HasDirectVirtualBases = false;
  bool HasNonVirtualBaseWithVBTable = false;

  // Now lay out the non-virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {

    // Ignore virtual bases, but remember that we saw one.
    if (I->isVirtual()) {
      HasDirectVirtualBases = true;
      continue;
    }

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->castAs<RecordType>()->getDecl());

    // Remember if this base has virtual bases itself.
    if (BaseDecl->getNumVBases())
      HasNonVirtualBaseWithVBTable = true;

    // Skip the primary base, because we've already laid it out.  The
    // !PrimaryBaseIsVirtual check is required because we might have a
    // non-virtual base of the same type as a primary virtual base.
    if (BaseDecl == PrimaryBase && !PrimaryBaseIsVirtual)
      continue;

    // Lay out the base.
    BaseSubobjectInfo *BaseInfo = NonVirtualBaseInfo.lookup(BaseDecl);
    assert(BaseInfo && "Did not find base info for non-virtual base!");

    LayoutNonVirtualBase(BaseInfo);
  }

  // In the MS ABI, add the vb-table pointer if we need one, which is
  // whenever we have a virtual base and we can't re-use a vb-table
  // pointer from a non-virtual base.
  if (isMicrosoftCXXABI() &&
      HasDirectVirtualBases && !HasNonVirtualBaseWithVBTable) {
    CharUnits PtrWidth = 
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerWidth(0));
    CharUnits PtrAlign = 
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerAlign(0));

    // MSVC potentially over-aligns the vb-table pointer by giving it
    // the max alignment of all the non-virtual objects in the class.
    // This is completely unnecessary, but we're not here to pass
    // judgment.
    //
    // Note that we've only laid out the non-virtual bases, so on the
    // first pass Alignment won't be set correctly here, but if the
    // vb-table doesn't end up aligned correctly we'll come through
    // and redo the layout from scratch with the right alignment.
    //
    // TODO: Instead of doing this, just lay out the fields as if the
    // vb-table were at offset zero, then retroactively bump the field
    // offsets up.
    PtrAlign = std::max(PtrAlign, Alignment);

    EnsureVTablePointerAlignment(PtrAlign);
    VBPtrOffset = getSize();
    setSize(getSize() + PtrWidth);
    setDataSize(getSize());
  }
}

void RecordLayoutBuilder::LayoutNonVirtualBase(const BaseSubobjectInfo *Base) {
  // Layout the base.
  CharUnits Offset = LayoutBase(Base);

  // Add its base class offset.
  assert(!Bases.count(Base->Class) && "base offset already exists!");
  Bases.insert(std::make_pair(Base->Class, Offset));

  AddPrimaryVirtualBaseOffsets(Base, Offset);
}

void
RecordLayoutBuilder::AddPrimaryVirtualBaseOffsets(const BaseSubobjectInfo *Info, 
                                                  CharUnits Offset) {
  // This base isn't interesting, it has no virtual bases.
  if (!Info->Class->getNumVBases())
    return;
  
  // First, check if we have a virtual primary base to add offsets for.
  if (Info->PrimaryVirtualBaseInfo) {
    assert(Info->PrimaryVirtualBaseInfo->IsVirtual && 
           "Primary virtual base is not virtual!");
    if (Info->PrimaryVirtualBaseInfo->Derived == Info) {
      // Add the offset.
      assert(!VBases.count(Info->PrimaryVirtualBaseInfo->Class) && 
             "primary vbase offset already exists!");
      VBases.insert(std::make_pair(Info->PrimaryVirtualBaseInfo->Class,
                                   ASTRecordLayout::VBaseInfo(Offset, false)));

      // Traverse the primary virtual base.
      AddPrimaryVirtualBaseOffsets(Info->PrimaryVirtualBaseInfo, Offset);
    }
  }

  // Now go through all direct non-virtual bases.
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(Info->Class);
  for (unsigned I = 0, E = Info->Bases.size(); I != E; ++I) {
    const BaseSubobjectInfo *Base = Info->Bases[I];
    if (Base->IsVirtual)
      continue;

    CharUnits BaseOffset = Offset + Layout.getBaseClassOffset(Base->Class);
    AddPrimaryVirtualBaseOffsets(Base, BaseOffset);
  }
}

/// needsVFTable - Return true if this class needs a vtable or vf-table
/// when laid out as a base class.  These are treated the same because
/// they're both always laid out at offset zero.
///
/// This function assumes that the class has no primary base.
bool RecordLayoutBuilder::needsVFTable(const CXXRecordDecl *RD) const {
  assert(!PrimaryBase);

  // In the Itanium ABI, every dynamic class needs a vtable: even if
  // this class has no virtual functions as a base class (i.e. it's
  // non-polymorphic or only has virtual functions from virtual
  // bases),x it still needs a vtable to locate its virtual bases.
  if (!isMicrosoftCXXABI())
    return RD->isDynamicClass();

  // In the MS ABI, we need a vfptr if the class has virtual functions
  // other than those declared by its virtual bases.  The AST doesn't
  // tell us that directly, and checking manually for virtual
  // functions that aren't overrides is expensive, but there are
  // some important shortcuts:

  //  - Non-polymorphic classes have no virtual functions at all.
  if (!RD->isPolymorphic()) return false;

  //  - Polymorphic classes with no virtual bases must either declare
  //    virtual functions directly or inherit them, but in the latter
  //    case we would have a primary base.
  if (RD->getNumVBases() == 0) return true;

  return hasNewVirtualFunction(RD);
}

/// Does the given class inherit non-virtually from any of the classes
/// in the given set?
static bool hasNonVirtualBaseInSet(const CXXRecordDecl *RD, 
                                   const ClassSetTy &set) {
  for (CXXRecordDecl::base_class_const_iterator
         I = RD->bases_begin(), E = RD->bases_end(); I != E; ++I) {
    // Ignore virtual links.
    if (I->isVirtual()) continue;

    // Check whether the set contains the base.
    const CXXRecordDecl *base = I->getType()->getAsCXXRecordDecl();
    if (set.count(base))
      return true;

    // Otherwise, recurse and propagate.
    if (hasNonVirtualBaseInSet(base, set))
      return true;
  }

  return false;
}

/// Does the given method (B::foo()) already override a method (A::foo())
/// such that A requires a vtordisp in B?  If so, we don't need to add a
/// new vtordisp for B in a yet-more-derived class C providing C::foo().
static bool overridesMethodRequiringVtorDisp(const ASTContext &Context,
                                             const CXXMethodDecl *M) {
  CXXMethodDecl::method_iterator
    I = M->begin_overridden_methods(), E = M->end_overridden_methods();
  if (I == E) return false;

  const ASTRecordLayout::VBaseOffsetsMapTy &offsets =
    Context.getASTRecordLayout(M->getParent()).getVBaseOffsetsMap();
  do {
    const CXXMethodDecl *overridden = *I;

    // If the overridden method's class isn't recognized as a virtual
    // base in the derived class, ignore it.
    ASTRecordLayout::VBaseOffsetsMapTy::const_iterator
      it = offsets.find(overridden->getParent());
    if (it == offsets.end()) continue;

    // Otherwise, check if the overridden method's class needs a vtordisp.
    if (it->second.hasVtorDisp()) return true;

  } while (++I != E);
  return false;
}                                             

/// In the Microsoft ABI, decide which of the virtual bases require a
/// vtordisp field.
void RecordLayoutBuilder::computeVtordisps(const CXXRecordDecl *RD,
                                           ClassSetTy &vtordispVBases) {
  // Bail out if we have no virtual bases.
  assert(RD->getNumVBases());

  // Build up the set of virtual bases that we haven't decided yet.
  ClassSetTy undecidedVBases;
  for (CXXRecordDecl::base_class_const_iterator
         I = RD->vbases_begin(), E = RD->vbases_end(); I != E; ++I) {
    const CXXRecordDecl *vbase = I->getType()->getAsCXXRecordDecl();
    undecidedVBases.insert(vbase);
  }
  assert(!undecidedVBases.empty());

  // A virtual base requires a vtordisp field in a derived class if it
  // requires a vtordisp field in a base class.  Walk all the direct
  // bases and collect this information.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    const CXXRecordDecl *base = I->getType()->getAsCXXRecordDecl();
    const ASTRecordLayout &baseLayout = Context.getASTRecordLayout(base);

    // Iterate over the set of virtual bases provided by this class.
    for (ASTRecordLayout::VBaseOffsetsMapTy::const_iterator
           VI = baseLayout.getVBaseOffsetsMap().begin(),
           VE = baseLayout.getVBaseOffsetsMap().end(); VI != VE; ++VI) {
      // If it doesn't need a vtordisp in this base, ignore it.
      if (!VI->second.hasVtorDisp()) continue;

      // If we've already seen it and decided it needs a vtordisp, ignore it.
      if (!undecidedVBases.erase(VI->first)) 
        continue;

      // Add it.
      vtordispVBases.insert(VI->first);

      // Quit as soon as we've decided everything.
      if (undecidedVBases.empty()) 
        return;
    }
  }

  // Okay, we have virtual bases that we haven't yet decided about.  A
  // virtual base requires a vtordisp if any the non-destructor
  // virtual methods declared in this class directly override a method
  // provided by that virtual base.  (If so, we need to emit a thunk
  // for that method, to be used in the construction vftable, which
  // applies an additional 'vtordisp' this-adjustment.)

  // Collect the set of bases directly overridden by any method in this class.
  // It's possible that some of these classes won't be virtual bases, or won't be
  // provided by virtual bases, or won't be virtual bases in the overridden
  // instance but are virtual bases elsewhere.  Only the last matters for what
  // we're doing, and we can ignore those:  if we don't directly override
  // a method provided by a virtual copy of a base class, but we do directly
  // override a method provided by a non-virtual copy of that base class,
  // then we must indirectly override the method provided by the virtual base,
  // and so we should already have collected it in the loop above.
  ClassSetTy overriddenBases;
  for (CXXRecordDecl::method_iterator
         M = RD->method_begin(), E = RD->method_end(); M != E; ++M) {
    // Ignore non-virtual methods and destructors.
    if (isa<CXXDestructorDecl>(*M) || !M->isVirtual())
      continue;
    
    for (CXXMethodDecl::method_iterator I = M->begin_overridden_methods(),
          E = M->end_overridden_methods(); I != E; ++I) {
      const CXXMethodDecl *overriddenMethod = (*I);

      // Ignore methods that override methods from vbases that require
      // require vtordisps.
      if (overridesMethodRequiringVtorDisp(Context, overriddenMethod))
        continue;

      // As an optimization, check immediately whether we're overriding
      // something from the undecided set.
      const CXXRecordDecl *overriddenBase = overriddenMethod->getParent();
      if (undecidedVBases.erase(overriddenBase)) {
        vtordispVBases.insert(overriddenBase);
        if (undecidedVBases.empty()) return;

        // We can't 'continue;' here because one of our undecided
        // vbases might non-virtually inherit from this base.
        // Consider:
        //   struct A { virtual void foo(); };
        //   struct B : A {};
        //   struct C : virtual A, virtual B { virtual void foo(); };
        // We need a vtordisp for B here.
      }

      // Otherwise, just collect it.
      overriddenBases.insert(overriddenBase);
    }
  }

  // Walk the undecided v-bases and check whether they (non-virtually)
  // provide any of the overridden bases.  We don't need to consider
  // virtual links because the vtordisp inheres to the layout
  // subobject containing the base.
  for (ClassSetTy::const_iterator
         I = undecidedVBases.begin(), E = undecidedVBases.end(); I != E; ++I) {
    if (hasNonVirtualBaseInSet(*I, overriddenBases))
      vtordispVBases.insert(*I);
  }
}

/// hasNewVirtualFunction - Does the given polymorphic class declare a
/// virtual function that does not override a method from any of its
/// base classes?
bool 
RecordLayoutBuilder::hasNewVirtualFunction(const CXXRecordDecl *RD, 
                                           bool IgnoreDestructor) const {
  if (!RD->getNumBases()) 
    return true;

  for (CXXRecordDecl::method_iterator method = RD->method_begin();
       method != RD->method_end();
       ++method) {
    if (method->isVirtual() && !method->size_overridden_methods() &&
        !(IgnoreDestructor && method->getKind() == Decl::CXXDestructor)) {
      return true;
    }
  }
  return false;
}

/// isPossiblePrimaryBase - Is the given base class an acceptable
/// primary base class?
bool 
RecordLayoutBuilder::isPossiblePrimaryBase(const CXXRecordDecl *base) const {
  // In the Itanium ABI, a class can be a primary base class if it has
  // a vtable for any reason.
  if (!isMicrosoftCXXABI())
    return base->isDynamicClass();

  // In the MS ABI, a class can only be a primary base class if it
  // provides a vf-table at a static offset.  That means it has to be
  // non-virtual base.  The existence of a separate vb-table means
  // that it's possible to get virtual functions only from a virtual
  // base, which we have to guard against.

  // First off, it has to have virtual functions.
  if (!base->isPolymorphic()) return false;

  // If it has no virtual bases, then the vfptr must be at a static offset.
  if (!base->getNumVBases()) return true;
  
  // Otherwise, the necessary information is cached in the layout.
  const ASTRecordLayout &layout = Context.getASTRecordLayout(base);

  // If the base has its own vfptr, it can be a primary base.
  if (layout.hasOwnVFPtr()) return true;

  // If the base has a primary base class, then it can be a primary base.
  if (layout.getPrimaryBase()) return true;

  // Otherwise it can't.
  return false;
}

void
RecordLayoutBuilder::LayoutVirtualBases(const CXXRecordDecl *RD,
                                        const CXXRecordDecl *MostDerivedClass) {
  const CXXRecordDecl *PrimaryBase;
  bool PrimaryBaseIsVirtual;

  if (MostDerivedClass == RD) {
    PrimaryBase = this->PrimaryBase;
    PrimaryBaseIsVirtual = this->PrimaryBaseIsVirtual;
  } else {
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    PrimaryBase = Layout.getPrimaryBase();
    PrimaryBaseIsVirtual = Layout.isPrimaryBaseVirtual();
  }

  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {
    assert(!I->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->castAs<RecordType>()->getDecl());

    if (I->isVirtual()) {
      if (PrimaryBase != BaseDecl || !PrimaryBaseIsVirtual) {
        bool IndirectPrimaryBase = IndirectPrimaryBases.count(BaseDecl);

        // Only lay out the virtual base if it's not an indirect primary base.
        if (!IndirectPrimaryBase) {
          // Only visit virtual bases once.
          if (!VisitedVirtualBases.insert(BaseDecl))
            continue;

          const BaseSubobjectInfo *BaseInfo = VirtualBaseInfo.lookup(BaseDecl);
          assert(BaseInfo && "Did not find virtual base info!");
          LayoutVirtualBase(BaseInfo);
        }
      }
    }

    if (!BaseDecl->getNumVBases()) {
      // This base isn't interesting since it doesn't have any virtual bases.
      continue;
    }

    LayoutVirtualBases(BaseDecl, MostDerivedClass);
  }
}

void RecordLayoutBuilder::MSLayoutVirtualBases(const CXXRecordDecl *RD) {
  if (!RD->getNumVBases())
    return;

  ClassSetTy VtordispVBases;
  computeVtordisps(RD, VtordispVBases);
  
  // This is substantially simplified because there are no virtual
  // primary bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->vbases_begin(),
       E = RD->vbases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl = I->getType()->getAsCXXRecordDecl();
    const BaseSubobjectInfo *BaseInfo = VirtualBaseInfo.lookup(BaseDecl);
    assert(BaseInfo && "Did not find virtual base info!");

    // If this base requires a vtordisp, add enough space for an int field.
    // This is apparently always 32-bits, even on x64.
    bool vtordispNeeded = false;
    if (VtordispVBases.count(BaseDecl)) {
      CharUnits IntSize = 
        CharUnits::fromQuantity(Context.getTargetInfo().getIntWidth() / 8);

      setSize(getSize() + IntSize);
      setDataSize(getSize());
      vtordispNeeded = true;
    }

    LayoutVirtualBase(BaseInfo, vtordispNeeded);
  }
}

void RecordLayoutBuilder::LayoutVirtualBase(const BaseSubobjectInfo *Base,
                                            bool IsVtordispNeed) {
  assert(!Base->Derived && "Trying to lay out a primary virtual base!");
  
  // Layout the base.
  CharUnits Offset = LayoutBase(Base);

  // Add its base class offset.
  assert(!VBases.count(Base->Class) && "vbase offset already exists!");
  VBases.insert(std::make_pair(Base->Class, 
                       ASTRecordLayout::VBaseInfo(Offset, IsVtordispNeed)));

  if (!isMicrosoftCXXABI())
    AddPrimaryVirtualBaseOffsets(Base, Offset);
}

CharUnits RecordLayoutBuilder::LayoutBase(const BaseSubobjectInfo *Base) {
  const ASTRecordLayout &Layout = Context.getASTRecordLayout(Base->Class);

  
  CharUnits Offset;
  
  // Query the external layout to see if it provides an offset.
  bool HasExternalLayout = false;
  if (ExternalLayout) {
    llvm::DenseMap<const CXXRecordDecl *, CharUnits>::iterator Known;
    if (Base->IsVirtual) {
      Known = ExternalVirtualBaseOffsets.find(Base->Class);
      if (Known != ExternalVirtualBaseOffsets.end()) {
        Offset = Known->second;
        HasExternalLayout = true;
      }
    } else {
      Known = ExternalBaseOffsets.find(Base->Class);
      if (Known != ExternalBaseOffsets.end()) {
        Offset = Known->second;
        HasExternalLayout = true;
      }
    }
  }
  
  // If we have an empty base class, try to place it at offset 0.
  if (Base->Class->isEmpty() &&
      (!HasExternalLayout || Offset == CharUnits::Zero()) &&
      EmptySubobjects->CanPlaceBaseAtOffset(Base, CharUnits::Zero())) {
    setSize(std::max(getSize(), Layout.getSize()));

    return CharUnits::Zero();
  }

  CharUnits UnpackedBaseAlign = Layout.getNonVirtualAlign();
  CharUnits BaseAlign = (Packed) ? CharUnits::One() : UnpackedBaseAlign;

  // The maximum field alignment overrides base align.
  if (!MaxFieldAlignment.isZero()) {
    BaseAlign = std::min(BaseAlign, MaxFieldAlignment);
    UnpackedBaseAlign = std::min(UnpackedBaseAlign, MaxFieldAlignment);
  }

  if (!HasExternalLayout) {
    // Round up the current record size to the base's alignment boundary.
    Offset = getDataSize().RoundUpToAlignment(BaseAlign);

    // Try to place the base.
    while (!EmptySubobjects->CanPlaceBaseAtOffset(Base, Offset))
      Offset += BaseAlign;
  } else {
    bool Allowed = EmptySubobjects->CanPlaceBaseAtOffset(Base, Offset);
    (void)Allowed;
    assert(Allowed && "Base subobject externally placed at overlapping offset");

    if (InferAlignment && Offset < getDataSize().RoundUpToAlignment(BaseAlign)){
      // The externally-supplied base offset is before the base offset we
      // computed. Assume that the structure is packed.
      Alignment = CharUnits::One();
      InferAlignment = false;
    }
  }
  
  if (!Base->Class->isEmpty()) {
    // Update the data size.
    setDataSize(Offset + Layout.getNonVirtualSize());

    setSize(std::max(getSize(), getDataSize()));
  } else
    setSize(std::max(getSize(), Offset + Layout.getSize()));

  // Remember max struct/class alignment.
  UpdateAlignment(BaseAlign, UnpackedBaseAlign);

  return Offset;
}

void RecordLayoutBuilder::InitializeLayout(const Decl *D) {
  if (const RecordDecl *RD = dyn_cast<RecordDecl>(D)) {
    IsUnion = RD->isUnion();
    IsMsStruct = RD->isMsStruct(Context);
  }

  Packed = D->hasAttr<PackedAttr>();  

  // Honor the default struct packing maximum alignment flag.
  if (unsigned DefaultMaxFieldAlignment = Context.getLangOpts().PackStruct) {
    MaxFieldAlignment = CharUnits::fromQuantity(DefaultMaxFieldAlignment);
  }

  // mac68k alignment supersedes maximum field alignment and attribute aligned,
  // and forces all structures to have 2-byte alignment. The IBM docs on it
  // allude to additional (more complicated) semantics, especially with regard
  // to bit-fields, but gcc appears not to follow that.
  if (D->hasAttr<AlignMac68kAttr>()) {
    IsMac68kAlign = true;
    MaxFieldAlignment = CharUnits::fromQuantity(2);
    Alignment = CharUnits::fromQuantity(2);
  } else {
    if (const MaxFieldAlignmentAttr *MFAA = D->getAttr<MaxFieldAlignmentAttr>())
      MaxFieldAlignment = Context.toCharUnitsFromBits(MFAA->getAlignment());

    if (unsigned MaxAlign = D->getMaxAlignment())
      UpdateAlignment(Context.toCharUnitsFromBits(MaxAlign));
  }
  
  // If there is an external AST source, ask it for the various offsets.
  if (const RecordDecl *RD = dyn_cast<RecordDecl>(D))
    if (ExternalASTSource *External = Context.getExternalSource()) {
      ExternalLayout = External->layoutRecordType(RD, 
                                                  ExternalSize,
                                                  ExternalAlign,
                                                  ExternalFieldOffsets,
                                                  ExternalBaseOffsets,
                                                  ExternalVirtualBaseOffsets);
      
      // Update based on external alignment.
      if (ExternalLayout) {
        if (ExternalAlign > 0) {
          Alignment = Context.toCharUnitsFromBits(ExternalAlign);
        } else {
          // The external source didn't have alignment information; infer it.
          InferAlignment = true;
        }
      }
    }
}

void RecordLayoutBuilder::Layout(const RecordDecl *D) {
  InitializeLayout(D);
  LayoutFields(D);

  // Finally, round the size of the total struct up to the alignment of the
  // struct itself.
  FinishLayout(D);
}

void RecordLayoutBuilder::Layout(const CXXRecordDecl *RD) {
  InitializeLayout(RD);

  // Lay out the vtable and the non-virtual bases.
  LayoutNonVirtualBases(RD);

  LayoutFields(RD);

  NonVirtualSize = Context.toCharUnitsFromBits(
        llvm::RoundUpToAlignment(getSizeInBits(), 
                                 Context.getTargetInfo().getCharAlign()));
  NonVirtualAlignment = Alignment;

  if (isMicrosoftCXXABI()) {
    if (NonVirtualSize != NonVirtualSize.RoundUpToAlignment(Alignment)) {
    CharUnits AlignMember = 
      NonVirtualSize.RoundUpToAlignment(Alignment) - NonVirtualSize;

    setSize(getSize() + AlignMember);
    setDataSize(getSize());

    NonVirtualSize = Context.toCharUnitsFromBits(
                             llvm::RoundUpToAlignment(getSizeInBits(),
                             Context.getTargetInfo().getCharAlign()));
    }

    MSLayoutVirtualBases(RD);
  } else {
    // Lay out the virtual bases and add the primary virtual base offsets.
    LayoutVirtualBases(RD, RD);
  }

  // Finally, round the size of the total struct up to the alignment
  // of the struct itself.
  FinishLayout(RD);

#ifndef NDEBUG
  // Check that we have base offsets for all bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
       E = RD->bases_end(); I != E; ++I) {
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    assert(Bases.count(BaseDecl) && "Did not find base offset!");
  }

  // And all virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->vbases_begin(),
       E = RD->vbases_end(); I != E; ++I) {
    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    assert(VBases.count(BaseDecl) && "Did not find base offset!");
  }
#endif
}

void RecordLayoutBuilder::Layout(const ObjCInterfaceDecl *D) {
  if (ObjCInterfaceDecl *SD = D->getSuperClass()) {
    const ASTRecordLayout &SL = Context.getASTObjCInterfaceLayout(SD);

    UpdateAlignment(SL.getAlignment());

    // We start laying out ivars not at the end of the superclass
    // structure, but at the next byte following the last field.
    setSize(SL.getDataSize());
    setDataSize(getSize());
  }

  InitializeLayout(D);
  // Layout each ivar sequentially.
  for (const ObjCIvarDecl *IVD = D->all_declared_ivar_begin(); IVD;
       IVD = IVD->getNextIvar())
    LayoutField(IVD);

  // Finally, round the size of the total struct up to the alignment of the
  // struct itself.
  FinishLayout(D);
}

void RecordLayoutBuilder::LayoutFields(const RecordDecl *D) {
  // Layout each field, for now, just sequentially, respecting alignment.  In
  // the future, this will need to be tweakable by targets.
  const FieldDecl *LastFD = 0;
  ZeroLengthBitfield = 0;
  unsigned RemainingInAlignment = 0;
  for (RecordDecl::field_iterator Field = D->field_begin(),
       FieldEnd = D->field_end(); Field != FieldEnd; ++Field) {
    if (IsMsStruct) {
      FieldDecl *FD = *Field;
      if (Context.ZeroBitfieldFollowsBitfield(FD, LastFD))
        ZeroLengthBitfield = FD;
      // Zero-length bitfields following non-bitfield members are
      // ignored:
      else if (Context.ZeroBitfieldFollowsNonBitfield(FD, LastFD))
        continue;
      // FIXME. streamline these conditions into a simple one.
      else if (Context.BitfieldFollowsBitfield(FD, LastFD) ||
               Context.BitfieldFollowsNonBitfield(FD, LastFD) ||
               Context.NonBitfieldFollowsBitfield(FD, LastFD)) {
        // 1) Adjacent bit fields are packed into the same 1-, 2-, or
        // 4-byte allocation unit if the integral types are the same
        // size and if the next bit field fits into the current
        // allocation unit without crossing the boundary imposed by the
        // common alignment requirements of the bit fields.
        // 2) Establish a new alignment for a bitfield following
        // a non-bitfield if size of their types differ.
        // 3) Establish a new alignment for a non-bitfield following
        // a bitfield if size of their types differ.
        std::pair<uint64_t, unsigned> FieldInfo = 
          Context.getTypeInfo(FD->getType());
        uint64_t TypeSize = FieldInfo.first;
        unsigned FieldAlign = FieldInfo.second;
        // This check is needed for 'long long' in -m32 mode.
        if (TypeSize > FieldAlign &&
            (Context.hasSameType(FD->getType(), 
                                Context.UnsignedLongLongTy) 
             ||Context.hasSameType(FD->getType(), 
                                   Context.LongLongTy)))
          FieldAlign = TypeSize;
        FieldInfo = Context.getTypeInfo(LastFD->getType());
        uint64_t TypeSizeLastFD = FieldInfo.first;
        unsigned FieldAlignLastFD = FieldInfo.second;
        // This check is needed for 'long long' in -m32 mode.
        if (TypeSizeLastFD > FieldAlignLastFD &&
            (Context.hasSameType(LastFD->getType(), 
                                Context.UnsignedLongLongTy)
             || Context.hasSameType(LastFD->getType(), 
                                    Context.LongLongTy)))
          FieldAlignLastFD = TypeSizeLastFD;
        
        if (TypeSizeLastFD != TypeSize) {
          if (RemainingInAlignment &&
              LastFD && LastFD->isBitField() &&
              LastFD->getBitWidthValue(Context)) {
            // If previous field was a bitfield with some remaining unfilled
            // bits, pad the field so current field starts on its type boundary.
            uint64_t FieldOffset = 
            getDataSizeInBits() - UnfilledBitsInLastByte;
            uint64_t NewSizeInBits = RemainingInAlignment + FieldOffset;
            setDataSize(llvm::RoundUpToAlignment(NewSizeInBits,
                                                 Context.getTargetInfo().getCharAlign()));
            setSize(std::max(getSizeInBits(), getDataSizeInBits()));
            RemainingInAlignment = 0;
          }
          
          uint64_t UnpaddedFieldOffset = 
            getDataSizeInBits() - UnfilledBitsInLastByte;
          FieldAlign = std::max(FieldAlign, FieldAlignLastFD);
          
          // The maximum field alignment overrides the aligned attribute.
          if (!MaxFieldAlignment.isZero()) {
            unsigned MaxFieldAlignmentInBits = 
              Context.toBits(MaxFieldAlignment);
            FieldAlign = std::min(FieldAlign, MaxFieldAlignmentInBits);
          }
          
          uint64_t NewSizeInBits = 
            llvm::RoundUpToAlignment(UnpaddedFieldOffset, FieldAlign);
          setDataSize(llvm::RoundUpToAlignment(NewSizeInBits,
                                               Context.getTargetInfo().getCharAlign()));
          UnfilledBitsInLastByte = getDataSizeInBits() - NewSizeInBits;
          setSize(std::max(getSizeInBits(), getDataSizeInBits()));
        }
        if (FD->isBitField()) {
          uint64_t FieldSize = FD->getBitWidthValue(Context);
          assert (FieldSize > 0 && "LayoutFields - ms_struct layout");
          if (RemainingInAlignment < FieldSize)
            RemainingInAlignment = TypeSize - FieldSize;
          else
            RemainingInAlignment -= FieldSize;
        }
      }
      else if (FD->isBitField()) {
        uint64_t FieldSize = FD->getBitWidthValue(Context);
        std::pair<uint64_t, unsigned> FieldInfo = 
          Context.getTypeInfo(FD->getType());
        uint64_t TypeSize = FieldInfo.first;
        RemainingInAlignment = TypeSize - FieldSize;
      }
      LastFD = FD;
    }
    else if (!Context.getTargetInfo().useBitFieldTypeAlignment() &&
             Context.getTargetInfo().useZeroLengthBitfieldAlignment()) {             
      if (Field->isBitField() && Field->getBitWidthValue(Context) == 0)
        ZeroLengthBitfield = *Field;
    }
    LayoutField(*Field);
  }
  if (IsMsStruct && RemainingInAlignment &&
      LastFD && LastFD->isBitField() && LastFD->getBitWidthValue(Context)) {
    // If we ended a bitfield before the full length of the type then
    // pad the struct out to the full length of the last type.
    uint64_t FieldOffset = 
      getDataSizeInBits() - UnfilledBitsInLastByte;
    uint64_t NewSizeInBits = RemainingInAlignment + FieldOffset;
    setDataSize(llvm::RoundUpToAlignment(NewSizeInBits,
                                         Context.getTargetInfo().getCharAlign()));
    setSize(std::max(getSizeInBits(), getDataSizeInBits()));
  }
}

void RecordLayoutBuilder::LayoutWideBitField(uint64_t FieldSize,
                                             uint64_t TypeSize,
                                             bool FieldPacked,
                                             const FieldDecl *D) {
  assert(Context.getLangOpts().CPlusPlus &&
         "Can only have wide bit-fields in C++!");

  // Itanium C++ ABI 2.4:
  //   If sizeof(T)*8 < n, let T' be the largest integral POD type with
  //   sizeof(T')*8 <= n.

  QualType IntegralPODTypes[] = {
    Context.UnsignedCharTy, Context.UnsignedShortTy, Context.UnsignedIntTy,
    Context.UnsignedLongTy, Context.UnsignedLongLongTy
  };

  QualType Type;
  for (unsigned I = 0, E = llvm::array_lengthof(IntegralPODTypes);
       I != E; ++I) {
    uint64_t Size = Context.getTypeSize(IntegralPODTypes[I]);

    if (Size > FieldSize)
      break;

    Type = IntegralPODTypes[I];
  }
  assert(!Type.isNull() && "Did not find a type!");

  CharUnits TypeAlign = Context.getTypeAlignInChars(Type);

  // We're not going to use any of the unfilled bits in the last byte.
  UnfilledBitsInLastByte = 0;

  uint64_t FieldOffset;
  uint64_t UnpaddedFieldOffset = getDataSizeInBits() - UnfilledBitsInLastByte;

  if (IsUnion) {
    setDataSize(std::max(getDataSizeInBits(), FieldSize));
    FieldOffset = 0;
  } else {
    // The bitfield is allocated starting at the next offset aligned 
    // appropriately for T', with length n bits.
    FieldOffset = llvm::RoundUpToAlignment(getDataSizeInBits(), 
                                           Context.toBits(TypeAlign));

    uint64_t NewSizeInBits = FieldOffset + FieldSize;

    setDataSize(llvm::RoundUpToAlignment(NewSizeInBits, 
                                         Context.getTargetInfo().getCharAlign()));
    UnfilledBitsInLastByte = getDataSizeInBits() - NewSizeInBits;
  }

  // Place this field at the current location.
  FieldOffsets.push_back(FieldOffset);

  CheckFieldPadding(FieldOffset, UnpaddedFieldOffset, FieldOffset,
                    Context.toBits(TypeAlign), FieldPacked, D);

  // Update the size.
  setSize(std::max(getSizeInBits(), getDataSizeInBits()));

  // Remember max struct/class alignment.
  UpdateAlignment(TypeAlign);
}

void RecordLayoutBuilder::LayoutBitField(const FieldDecl *D) {
  bool FieldPacked = Packed || D->hasAttr<PackedAttr>();
  uint64_t UnpaddedFieldOffset = getDataSizeInBits() - UnfilledBitsInLastByte;
  uint64_t FieldOffset = IsUnion ? 0 : UnpaddedFieldOffset;
  uint64_t FieldSize = D->getBitWidthValue(Context);

  std::pair<uint64_t, unsigned> FieldInfo = Context.getTypeInfo(D->getType());
  uint64_t TypeSize = FieldInfo.first;
  unsigned FieldAlign = FieldInfo.second;
  
  // This check is needed for 'long long' in -m32 mode.
  if (IsMsStruct && (TypeSize > FieldAlign) && 
      (Context.hasSameType(D->getType(), 
                           Context.UnsignedLongLongTy) 
       || Context.hasSameType(D->getType(), Context.LongLongTy)))
    FieldAlign = TypeSize;

  if (ZeroLengthBitfield) {
    std::pair<uint64_t, unsigned> FieldInfo;
    unsigned ZeroLengthBitfieldAlignment;
    if (IsMsStruct) {
      // If a zero-length bitfield is inserted after a bitfield,
      // and the alignment of the zero-length bitfield is
      // greater than the member that follows it, `bar', `bar' 
      // will be aligned as the type of the zero-length bitfield.
      if (ZeroLengthBitfield != D) {
        FieldInfo = Context.getTypeInfo(ZeroLengthBitfield->getType());
        ZeroLengthBitfieldAlignment = FieldInfo.second;
        // Ignore alignment of subsequent zero-length bitfields.
        if ((ZeroLengthBitfieldAlignment > FieldAlign) || (FieldSize == 0))
          FieldAlign = ZeroLengthBitfieldAlignment;
        if (FieldSize)
          ZeroLengthBitfield = 0;
      }
    } else {
      // The alignment of a zero-length bitfield affects the alignment
      // of the next member.  The alignment is the max of the zero 
      // length bitfield's alignment and a target specific fixed value.
      unsigned ZeroLengthBitfieldBoundary =
        Context.getTargetInfo().getZeroLengthBitfieldBoundary();
      if (ZeroLengthBitfieldBoundary > FieldAlign)
        FieldAlign = ZeroLengthBitfieldBoundary;
    }
  }

  if (FieldSize > TypeSize) {
    LayoutWideBitField(FieldSize, TypeSize, FieldPacked, D);
    return;
  }

  // The align if the field is not packed. This is to check if the attribute
  // was unnecessary (-Wpacked).
  unsigned UnpackedFieldAlign = FieldAlign;
  uint64_t UnpackedFieldOffset = FieldOffset;
  if (!Context.getTargetInfo().useBitFieldTypeAlignment() && !ZeroLengthBitfield)
    UnpackedFieldAlign = 1;

  if (FieldPacked || 
      (!Context.getTargetInfo().useBitFieldTypeAlignment() && !ZeroLengthBitfield))
    FieldAlign = 1;
  FieldAlign = std::max(FieldAlign, D->getMaxAlignment());
  UnpackedFieldAlign = std::max(UnpackedFieldAlign, D->getMaxAlignment());

  // The maximum field alignment overrides the aligned attribute.
  if (!MaxFieldAlignment.isZero() && FieldSize != 0) {
    unsigned MaxFieldAlignmentInBits = Context.toBits(MaxFieldAlignment);
    FieldAlign = std::min(FieldAlign, MaxFieldAlignmentInBits);
    UnpackedFieldAlign = std::min(UnpackedFieldAlign, MaxFieldAlignmentInBits);
  }

  // Check if we need to add padding to give the field the correct alignment.
  if (FieldSize == 0 || 
      (MaxFieldAlignment.isZero() &&
       (FieldOffset & (FieldAlign-1)) + FieldSize > TypeSize))
    FieldOffset = llvm::RoundUpToAlignment(FieldOffset, FieldAlign);

  if (FieldSize == 0 ||
      (MaxFieldAlignment.isZero() &&
       (UnpackedFieldOffset & (UnpackedFieldAlign-1)) + FieldSize > TypeSize))
    UnpackedFieldOffset = llvm::RoundUpToAlignment(UnpackedFieldOffset,
                                                   UnpackedFieldAlign);

  // Padding members don't affect overall alignment, unless zero length bitfield
  // alignment is enabled.
  if (!D->getIdentifier() && !Context.getTargetInfo().useZeroLengthBitfieldAlignment())
    FieldAlign = UnpackedFieldAlign = 1;

  if (!IsMsStruct)
    ZeroLengthBitfield = 0;

  if (ExternalLayout)
    FieldOffset = updateExternalFieldOffset(D, FieldOffset);

  // Place this field at the current location.
  FieldOffsets.push_back(FieldOffset);

  if (!ExternalLayout)
    CheckFieldPadding(FieldOffset, UnpaddedFieldOffset, UnpackedFieldOffset,
                      UnpackedFieldAlign, FieldPacked, D);

  // Update DataSize to include the last byte containing (part of) the bitfield.
  if (IsUnion) {
    // FIXME: I think FieldSize should be TypeSize here.
    setDataSize(std::max(getDataSizeInBits(), FieldSize));
  } else {
    uint64_t NewSizeInBits = FieldOffset + FieldSize;

    setDataSize(llvm::RoundUpToAlignment(NewSizeInBits, 
                                         Context.getTargetInfo().getCharAlign()));
    UnfilledBitsInLastByte = getDataSizeInBits() - NewSizeInBits;
  }

  // Update the size.
  setSize(std::max(getSizeInBits(), getDataSizeInBits()));

  // Remember max struct/class alignment.
  UpdateAlignment(Context.toCharUnitsFromBits(FieldAlign), 
                  Context.toCharUnitsFromBits(UnpackedFieldAlign));
}

void RecordLayoutBuilder::LayoutField(const FieldDecl *D) {  
  if (D->isBitField()) {
    LayoutBitField(D);
    return;
  }

  uint64_t UnpaddedFieldOffset = getDataSizeInBits() - UnfilledBitsInLastByte;

  // Reset the unfilled bits.
  UnfilledBitsInLastByte = 0;

  bool FieldPacked = Packed || D->hasAttr<PackedAttr>();
  CharUnits FieldOffset = 
    IsUnion ? CharUnits::Zero() : getDataSize();
  CharUnits FieldSize;
  CharUnits FieldAlign;

  if (D->getType()->isIncompleteArrayType()) {
    // This is a flexible array member; we can't directly
    // query getTypeInfo about these, so we figure it out here.
    // Flexible array members don't have any size, but they
    // have to be aligned appropriately for their element type.
    FieldSize = CharUnits::Zero();
    const ArrayType* ATy = Context.getAsArrayType(D->getType());
    FieldAlign = Context.getTypeAlignInChars(ATy->getElementType());
  } else if (const ReferenceType *RT = D->getType()->getAs<ReferenceType>()) {
    unsigned AS = RT->getPointeeType().getAddressSpace();
    FieldSize = 
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerWidth(AS));
    FieldAlign = 
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerAlign(AS));
  } else {
    std::pair<CharUnits, CharUnits> FieldInfo = 
      Context.getTypeInfoInChars(D->getType());
    FieldSize = FieldInfo.first;
    FieldAlign = FieldInfo.second;

    if (ZeroLengthBitfield) {
      CharUnits ZeroLengthBitfieldBoundary = 
        Context.toCharUnitsFromBits(
          Context.getTargetInfo().getZeroLengthBitfieldBoundary());
      if (ZeroLengthBitfieldBoundary == CharUnits::Zero()) {
        // If a zero-length bitfield is inserted after a bitfield,
        // and the alignment of the zero-length bitfield is
        // greater than the member that follows it, `bar', `bar' 
        // will be aligned as the type of the zero-length bitfield.
        std::pair<CharUnits, CharUnits> FieldInfo = 
          Context.getTypeInfoInChars(ZeroLengthBitfield->getType());
        CharUnits ZeroLengthBitfieldAlignment = FieldInfo.second;        
        if (ZeroLengthBitfieldAlignment > FieldAlign)
          FieldAlign = ZeroLengthBitfieldAlignment;
      } else if (ZeroLengthBitfieldBoundary > FieldAlign) {
        // Align 'bar' based on a fixed alignment specified by the target.
        assert(Context.getTargetInfo().useZeroLengthBitfieldAlignment() &&
               "ZeroLengthBitfieldBoundary should only be used in conjunction"
               " with useZeroLengthBitfieldAlignment.");
        FieldAlign = ZeroLengthBitfieldBoundary;
      }
      ZeroLengthBitfield = 0;
    }

    if (IsMsStruct) {
      // If MS bitfield layout is required, figure out what type is being
      // laid out and align the field to the width of that type.
      
      // Resolve all typedefs down to their base type and round up the field
      // alignment if necessary.
      QualType T = Context.getBaseElementType(D->getType());
      if (const BuiltinType *BTy = T->getAs<BuiltinType>()) {
        CharUnits TypeSize = Context.getTypeSizeInChars(BTy);
        if (TypeSize > FieldAlign)
          FieldAlign = TypeSize;
      }
    }
  }

  // The align if the field is not packed. This is to check if the attribute
  // was unnecessary (-Wpacked).
  CharUnits UnpackedFieldAlign = FieldAlign;
  CharUnits UnpackedFieldOffset = FieldOffset;

  if (FieldPacked)
    FieldAlign = CharUnits::One();
  CharUnits MaxAlignmentInChars = 
    Context.toCharUnitsFromBits(D->getMaxAlignment());
  FieldAlign = std::max(FieldAlign, MaxAlignmentInChars);
  UnpackedFieldAlign = std::max(UnpackedFieldAlign, MaxAlignmentInChars);

  // The maximum field alignment overrides the aligned attribute.
  if (!MaxFieldAlignment.isZero()) {
    FieldAlign = std::min(FieldAlign, MaxFieldAlignment);
    UnpackedFieldAlign = std::min(UnpackedFieldAlign, MaxFieldAlignment);
  }

  // Round up the current record size to the field's alignment boundary.
  FieldOffset = FieldOffset.RoundUpToAlignment(FieldAlign);
  UnpackedFieldOffset = 
    UnpackedFieldOffset.RoundUpToAlignment(UnpackedFieldAlign);

  if (ExternalLayout) {
    FieldOffset = Context.toCharUnitsFromBits(
                    updateExternalFieldOffset(D, Context.toBits(FieldOffset)));
    
    if (!IsUnion && EmptySubobjects) {
      // Record the fact that we're placing a field at this offset.
      bool Allowed = EmptySubobjects->CanPlaceFieldAtOffset(D, FieldOffset);
      (void)Allowed;
      assert(Allowed && "Externally-placed field cannot be placed here");      
    }
  } else {
    if (!IsUnion && EmptySubobjects) {
      // Check if we can place the field at this offset.
      while (!EmptySubobjects->CanPlaceFieldAtOffset(D, FieldOffset)) {
        // We couldn't place the field at the offset. Try again at a new offset.
        FieldOffset += FieldAlign;
      }
    }
  }
  
  // Place this field at the current location.
  FieldOffsets.push_back(Context.toBits(FieldOffset));

  if (!ExternalLayout)
    CheckFieldPadding(Context.toBits(FieldOffset), UnpaddedFieldOffset, 
                      Context.toBits(UnpackedFieldOffset),
                      Context.toBits(UnpackedFieldAlign), FieldPacked, D);

  // Reserve space for this field.
  uint64_t FieldSizeInBits = Context.toBits(FieldSize);
  if (IsUnion)
    setDataSize(std::max(getDataSizeInBits(), FieldSizeInBits));
  else
    setDataSize(FieldOffset + FieldSize);

  // Update the size.
  setSize(std::max(getSizeInBits(), getDataSizeInBits()));

  // Remember max struct/class alignment.
  UpdateAlignment(FieldAlign, UnpackedFieldAlign);
}

void RecordLayoutBuilder::FinishLayout(const NamedDecl *D) {
  // In C++, records cannot be of size 0.
  if (Context.getLangOpts().CPlusPlus && getSizeInBits() == 0) {
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D)) {
      // Compatibility with gcc requires a class (pod or non-pod)
      // which is not empty but of size 0; such as having fields of
      // array of zero-length, remains of Size 0
      if (RD->isEmpty())
        setSize(CharUnits::One());
    }
    else
      setSize(CharUnits::One());
  }

  // Finally, round the size of the record up to the alignment of the
  // record itself.
  uint64_t UnpaddedSize = getSizeInBits() - UnfilledBitsInLastByte;
  uint64_t UnpackedSizeInBits =
  llvm::RoundUpToAlignment(getSizeInBits(),
                           Context.toBits(UnpackedAlignment));
  CharUnits UnpackedSize = Context.toCharUnitsFromBits(UnpackedSizeInBits);
  uint64_t RoundedSize
    = llvm::RoundUpToAlignment(getSizeInBits(), Context.toBits(Alignment));

  if (ExternalLayout) {
    // If we're inferring alignment, and the external size is smaller than
    // our size after we've rounded up to alignment, conservatively set the
    // alignment to 1.
    if (InferAlignment && ExternalSize < RoundedSize) {
      Alignment = CharUnits::One();
      InferAlignment = false;
    }
    setSize(ExternalSize);
    return;
  }


  // MSVC doesn't round up to the alignment of the record with virtual bases.
  if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D)) {
    if (isMicrosoftCXXABI() && RD->getNumVBases())
      return;
  }

  // Set the size to the final size.
  setSize(RoundedSize);

  unsigned CharBitNum = Context.getTargetInfo().getCharWidth();
  if (const RecordDecl *RD = dyn_cast<RecordDecl>(D)) {
    // Warn if padding was introduced to the struct/class/union.
    if (getSizeInBits() > UnpaddedSize) {
      unsigned PadSize = getSizeInBits() - UnpaddedSize;
      bool InBits = true;
      if (PadSize % CharBitNum == 0) {
        PadSize = PadSize / CharBitNum;
        InBits = false;
      }
      Diag(RD->getLocation(), diag::warn_padded_struct_size)
          << Context.getTypeDeclType(RD)
          << PadSize
          << (InBits ? 1 : 0) /*(byte|bit)*/ << (PadSize > 1); // plural or not
    }

    // Warn if we packed it unnecessarily. If the alignment is 1 byte don't
    // bother since there won't be alignment issues.
    if (Packed && UnpackedAlignment > CharUnits::One() && 
        getSize() == UnpackedSize)
      Diag(D->getLocation(), diag::warn_unnecessary_packed)
          << Context.getTypeDeclType(RD);
  }
}

void RecordLayoutBuilder::UpdateAlignment(CharUnits NewAlignment,
                                          CharUnits UnpackedNewAlignment) {
  // The alignment is not modified when using 'mac68k' alignment or when
  // we have an externally-supplied layout that also provides overall alignment.
  if (IsMac68kAlign || (ExternalLayout && !InferAlignment))
    return;

  if (NewAlignment > Alignment) {
    assert(llvm::isPowerOf2_32(NewAlignment.getQuantity() && 
           "Alignment not a power of 2"));
    Alignment = NewAlignment;
  }

  if (UnpackedNewAlignment > UnpackedAlignment) {
    assert(llvm::isPowerOf2_32(UnpackedNewAlignment.getQuantity() &&
           "Alignment not a power of 2"));
    UnpackedAlignment = UnpackedNewAlignment;
  }
}

uint64_t
RecordLayoutBuilder::updateExternalFieldOffset(const FieldDecl *Field, 
                                               uint64_t ComputedOffset) {
  assert(ExternalFieldOffsets.find(Field) != ExternalFieldOffsets.end() &&
         "Field does not have an external offset");
  
  uint64_t ExternalFieldOffset = ExternalFieldOffsets[Field];
  
  if (InferAlignment && ExternalFieldOffset < ComputedOffset) {
    // The externally-supplied field offset is before the field offset we
    // computed. Assume that the structure is packed.
    Alignment = CharUnits::One();
    InferAlignment = false;
  }
  
  // Use the externally-supplied field offset.
  return ExternalFieldOffset;
}

/// \brief Get diagnostic %select index for tag kind for
/// field padding diagnostic message.
/// WARNING: Indexes apply to particular diagnostics only!
///
/// \returns diagnostic %select index.
static unsigned getPaddingDiagFromTagKind(TagTypeKind Tag) {
  switch (Tag) {
  case TTK_Struct: return 0;
  case TTK_Interface: return 1;
  case TTK_Class: return 2;
  default: llvm_unreachable("Invalid tag kind for field padding diagnostic!");
  }
}

void RecordLayoutBuilder::CheckFieldPadding(uint64_t Offset,
                                            uint64_t UnpaddedOffset,
                                            uint64_t UnpackedOffset,
                                            unsigned UnpackedAlign,
                                            bool isPacked,
                                            const FieldDecl *D) {
  // We let objc ivars without warning, objc interfaces generally are not used
  // for padding tricks.
  if (isa<ObjCIvarDecl>(D))
    return;

  // Don't warn about structs created without a SourceLocation.  This can
  // be done by clients of the AST, such as codegen.
  if (D->getLocation().isInvalid())
    return;
  
  unsigned CharBitNum = Context.getTargetInfo().getCharWidth();

  // Warn if padding was introduced to the struct/class.
  if (!IsUnion && Offset > UnpaddedOffset) {
    unsigned PadSize = Offset - UnpaddedOffset;
    bool InBits = true;
    if (PadSize % CharBitNum == 0) {
      PadSize = PadSize / CharBitNum;
      InBits = false;
    }
    if (D->getIdentifier())
      Diag(D->getLocation(), diag::warn_padded_struct_field)
          << getPaddingDiagFromTagKind(D->getParent()->getTagKind())
          << Context.getTypeDeclType(D->getParent())
          << PadSize
          << (InBits ? 1 : 0) /*(byte|bit)*/ << (PadSize > 1) // plural or not
          << D->getIdentifier();
    else
      Diag(D->getLocation(), diag::warn_padded_struct_anon_field)
          << getPaddingDiagFromTagKind(D->getParent()->getTagKind())
          << Context.getTypeDeclType(D->getParent())
          << PadSize
          << (InBits ? 1 : 0) /*(byte|bit)*/ << (PadSize > 1); // plural or not
  }

  // Warn if we packed it unnecessarily. If the alignment is 1 byte don't
  // bother since there won't be alignment issues.
  if (isPacked && UnpackedAlign > CharBitNum && Offset == UnpackedOffset)
    Diag(D->getLocation(), diag::warn_unnecessary_packed)
        << D->getIdentifier();
}

const CXXMethodDecl *
RecordLayoutBuilder::ComputeKeyFunction(const CXXRecordDecl *RD) {
  // If a class isn't polymorphic it doesn't have a key function.
  if (!RD->isPolymorphic())
    return 0;

  // A class that is not externally visible doesn't have a key function. (Or
  // at least, there's no point to assigning a key function to such a class;
  // this doesn't affect the ABI.)
  if (RD->getLinkage() != ExternalLinkage)
    return 0;

  // Template instantiations don't have key functions,see Itanium C++ ABI 5.2.6.
  // Same behavior as GCC.
  TemplateSpecializationKind TSK = RD->getTemplateSpecializationKind();
  if (TSK == TSK_ImplicitInstantiation ||
      TSK == TSK_ExplicitInstantiationDefinition)
    return 0;

  for (CXXRecordDecl::method_iterator I = RD->method_begin(),
         E = RD->method_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;

    if (!MD->isVirtual())
      continue;

    if (MD->isPure())
      continue;

    // Ignore implicit member functions, they are always marked as inline, but
    // they don't have a body until they're defined.
    if (MD->isImplicit())
      continue;

    if (MD->isInlineSpecified())
      continue;

    if (MD->hasInlineBody())
      continue;

    // Ignore inline deleted or defaulted functions.
    if (!MD->isUserProvided())
      continue;

    // We found it.
    return MD;
  }

  return 0;
}

DiagnosticBuilder
RecordLayoutBuilder::Diag(SourceLocation Loc, unsigned DiagID) {
  return Context.getDiagnostics().Report(Loc, DiagID);
}

/// getASTRecordLayout - Get or compute information about the layout of the
/// specified record (struct/union/class), which indicates its size and field
/// position information.
const ASTRecordLayout &
ASTContext::getASTRecordLayout(const RecordDecl *D) const {
  // These asserts test different things.  A record has a definition
  // as soon as we begin to parse the definition.  That definition is
  // not a complete definition (which is what isDefinition() tests)
  // until we *finish* parsing the definition.

  if (D->hasExternalLexicalStorage() && !D->getDefinition())
    getExternalSource()->CompleteType(const_cast<RecordDecl*>(D));
    
  D = D->getDefinition();
  assert(D && "Cannot get layout of forward declarations!");
  assert(D->isCompleteDefinition() && "Cannot layout type before complete!");

  // Look up this layout, if already laid out, return what we have.
  // Note that we can't save a reference to the entry because this function
  // is recursive.
  const ASTRecordLayout *Entry = ASTRecordLayouts[D];
  if (Entry) return *Entry;

  const ASTRecordLayout *NewEntry;

  if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D)) {
    EmptySubobjectMap EmptySubobjects(*this, RD);
    RecordLayoutBuilder Builder(*this, &EmptySubobjects);
    Builder.Layout(RD);

    // MSVC gives the vb-table pointer an alignment equal to that of
    // the non-virtual part of the structure.  That's an inherently
    // multi-pass operation.  If our first pass doesn't give us
    // adequate alignment, try again with the specified minimum
    // alignment.  This is *much* more maintainable than computing the
    // alignment in advance in a separately-coded pass; it's also
    // significantly more efficient in the common case where the
    // vb-table doesn't need extra padding.
    if (Builder.VBPtrOffset != CharUnits::fromQuantity(-1) &&
        (Builder.VBPtrOffset % Builder.NonVirtualAlignment) != 0) {
      Builder.resetWithTargetAlignment(Builder.NonVirtualAlignment);
      Builder.Layout(RD);
    }

    // FIXME: This is not always correct. See the part about bitfields at
    // http://www.codesourcery.com/public/cxx-abi/abi.html#POD for more info.
    // FIXME: IsPODForThePurposeOfLayout should be stored in the record layout.
    // This does not affect the calculations of MSVC layouts
    bool IsPODForThePurposeOfLayout = 
      (!Builder.isMicrosoftCXXABI() && cast<CXXRecordDecl>(D)->isPOD());

    // FIXME: This should be done in FinalizeLayout.
    CharUnits DataSize =
      IsPODForThePurposeOfLayout ? Builder.getSize() : Builder.getDataSize();
    CharUnits NonVirtualSize = 
      IsPODForThePurposeOfLayout ? DataSize : Builder.NonVirtualSize;

    NewEntry =
      new (*this) ASTRecordLayout(*this, Builder.getSize(), 
                                  Builder.Alignment,
                                  Builder.HasOwnVFPtr,
                                  Builder.VBPtrOffset,
                                  DataSize, 
                                  Builder.FieldOffsets.data(),
                                  Builder.FieldOffsets.size(),
                                  NonVirtualSize,
                                  Builder.NonVirtualAlignment,
                                  EmptySubobjects.SizeOfLargestEmptySubobject,
                                  Builder.PrimaryBase,
                                  Builder.PrimaryBaseIsVirtual,
                                  Builder.Bases, Builder.VBases);
  } else {
    RecordLayoutBuilder Builder(*this, /*EmptySubobjects=*/0);
    Builder.Layout(D);

    NewEntry =
      new (*this) ASTRecordLayout(*this, Builder.getSize(), 
                                  Builder.Alignment,
                                  Builder.getSize(),
                                  Builder.FieldOffsets.data(),
                                  Builder.FieldOffsets.size());
  }

  ASTRecordLayouts[D] = NewEntry;

  if (getLangOpts().DumpRecordLayouts) {
    llvm::errs() << "\n*** Dumping AST Record Layout\n";
    DumpRecordLayout(D, llvm::errs(), getLangOpts().DumpRecordLayoutsSimple);
  }

  return *NewEntry;
}

const CXXMethodDecl *ASTContext::getKeyFunction(const CXXRecordDecl *RD) {
  RD = cast<CXXRecordDecl>(RD->getDefinition());
  assert(RD && "Cannot get key function for forward declarations!");

  const CXXMethodDecl *&Entry = KeyFunctions[RD];
  if (!Entry)
    Entry = RecordLayoutBuilder::ComputeKeyFunction(RD);

  return Entry;
}

static uint64_t getFieldOffset(const ASTContext &C, const FieldDecl *FD) {
  const ASTRecordLayout &Layout = C.getASTRecordLayout(FD->getParent());
  return Layout.getFieldOffset(FD->getFieldIndex());
}

uint64_t ASTContext::getFieldOffset(const ValueDecl *VD) const {
  uint64_t OffsetInBits;
  if (const FieldDecl *FD = dyn_cast<FieldDecl>(VD)) {
    OffsetInBits = ::getFieldOffset(*this, FD);
  } else {
    const IndirectFieldDecl *IFD = cast<IndirectFieldDecl>(VD);

    OffsetInBits = 0;
    for (IndirectFieldDecl::chain_iterator CI = IFD->chain_begin(),
                                           CE = IFD->chain_end();
         CI != CE; ++CI)
      OffsetInBits += ::getFieldOffset(*this, cast<FieldDecl>(*CI));
  }

  return OffsetInBits;
}

/// getObjCLayout - Get or compute information about the layout of the
/// given interface.
///
/// \param Impl - If given, also include the layout of the interface's
/// implementation. This may differ by including synthesized ivars.
const ASTRecordLayout &
ASTContext::getObjCLayout(const ObjCInterfaceDecl *D,
                          const ObjCImplementationDecl *Impl) const {
  // Retrieve the definition
  if (D->hasExternalLexicalStorage() && !D->getDefinition())
    getExternalSource()->CompleteType(const_cast<ObjCInterfaceDecl*>(D));
  D = D->getDefinition();
  assert(D && D->isThisDeclarationADefinition() && "Invalid interface decl!");

  // Look up this layout, if already laid out, return what we have.
  const ObjCContainerDecl *Key =
    Impl ? (const ObjCContainerDecl*) Impl : (const ObjCContainerDecl*) D;
  if (const ASTRecordLayout *Entry = ObjCLayouts[Key])
    return *Entry;

  // Add in synthesized ivar count if laying out an implementation.
  if (Impl) {
    unsigned SynthCount = CountNonClassIvars(D);
    // If there aren't any sythesized ivars then reuse the interface
    // entry. Note we can't cache this because we simply free all
    // entries later; however we shouldn't look up implementations
    // frequently.
    if (SynthCount == 0)
      return getObjCLayout(D, 0);
  }

  RecordLayoutBuilder Builder(*this, /*EmptySubobjects=*/0);
  Builder.Layout(D);

  const ASTRecordLayout *NewEntry =
    new (*this) ASTRecordLayout(*this, Builder.getSize(), 
                                Builder.Alignment,
                                Builder.getDataSize(),
                                Builder.FieldOffsets.data(),
                                Builder.FieldOffsets.size());

  ObjCLayouts[Key] = NewEntry;

  return *NewEntry;
}

static void PrintOffset(raw_ostream &OS,
                        CharUnits Offset, unsigned IndentLevel) {
  OS << llvm::format("%4" PRId64 " | ", (int64_t)Offset.getQuantity());
  OS.indent(IndentLevel * 2);
}

static void DumpCXXRecordLayout(raw_ostream &OS,
                                const CXXRecordDecl *RD, const ASTContext &C,
                                CharUnits Offset,
                                unsigned IndentLevel,
                                const char* Description,
                                bool IncludeVirtualBases) {
  const ASTRecordLayout &Layout = C.getASTRecordLayout(RD);

  PrintOffset(OS, Offset, IndentLevel);
  OS << C.getTypeDeclType(const_cast<CXXRecordDecl *>(RD)).getAsString();
  if (Description)
    OS << ' ' << Description;
  if (RD->isEmpty())
    OS << " (empty)";
  OS << '\n';

  IndentLevel++;

  const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
  bool HasVfptr = Layout.hasOwnVFPtr();
  bool HasVbptr = Layout.getVBPtrOffset() != CharUnits::fromQuantity(-1);

  // Vtable pointer.
  if (RD->isDynamicClass() && !PrimaryBase &&
      C.getTargetInfo().getCXXABI() != CXXABI_Microsoft) {
    PrintOffset(OS, Offset, IndentLevel);
    OS << '(' << *RD << " vtable pointer)\n";
  }
  
  // Dump (non-virtual) bases
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {
    assert(!I->getType()->isDependentType() &&
           "Cannot layout class with dependent bases.");
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *Base =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    CharUnits BaseOffset = Offset + Layout.getBaseClassOffset(Base);

    DumpCXXRecordLayout(OS, Base, C, BaseOffset, IndentLevel,
                        Base == PrimaryBase ? "(primary base)" : "(base)",
                        /*IncludeVirtualBases=*/false);
  }

  // vfptr and vbptr (for Microsoft C++ ABI)
  if (HasVfptr) {
    PrintOffset(OS, Offset, IndentLevel);
    OS << '(' << *RD << " vftable pointer)\n";
  }
  if (HasVbptr) {
    PrintOffset(OS, Offset + Layout.getVBPtrOffset(), IndentLevel);
    OS << '(' << *RD << " vbtable pointer)\n";
  }

  // Dump fields.
  uint64_t FieldNo = 0;
  for (CXXRecordDecl::field_iterator I = RD->field_begin(),
         E = RD->field_end(); I != E; ++I, ++FieldNo) {
    const FieldDecl &Field = **I;
    CharUnits FieldOffset = Offset + 
      C.toCharUnitsFromBits(Layout.getFieldOffset(FieldNo));

    if (const RecordType *RT = Field.getType()->getAs<RecordType>()) {
      if (const CXXRecordDecl *D = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
        DumpCXXRecordLayout(OS, D, C, FieldOffset, IndentLevel,
                            Field.getName().data(),
                            /*IncludeVirtualBases=*/true);
        continue;
      }
    }

    PrintOffset(OS, FieldOffset, IndentLevel);
    OS << Field.getType().getAsString() << ' ' << Field << '\n';
  }

  if (!IncludeVirtualBases)
    return;

  // Dump virtual bases.
  const ASTRecordLayout::VBaseOffsetsMapTy &vtordisps = 
    Layout.getVBaseOffsetsMap();
  for (CXXRecordDecl::base_class_const_iterator I = RD->vbases_begin(),
         E = RD->vbases_end(); I != E; ++I) {
    assert(I->isVirtual() && "Found non-virtual class!");
    const CXXRecordDecl *VBase =
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());

    CharUnits VBaseOffset = Offset + Layout.getVBaseClassOffset(VBase);

    if (vtordisps.find(VBase)->second.hasVtorDisp()) {
      PrintOffset(OS, VBaseOffset - CharUnits::fromQuantity(4), IndentLevel);
      OS << "(vtordisp for vbase " << *VBase << ")\n";
    }

    DumpCXXRecordLayout(OS, VBase, C, VBaseOffset, IndentLevel,
                        VBase == PrimaryBase ?
                        "(primary virtual base)" : "(virtual base)",
                        /*IncludeVirtualBases=*/false);
  }

  OS << "  sizeof=" << Layout.getSize().getQuantity();
  OS << ", dsize=" << Layout.getDataSize().getQuantity();
  OS << ", align=" << Layout.getAlignment().getQuantity() << '\n';
  OS << "  nvsize=" << Layout.getNonVirtualSize().getQuantity();
  OS << ", nvalign=" << Layout.getNonVirtualAlign().getQuantity() << '\n';
  OS << '\n';
}

void ASTContext::DumpRecordLayout(const RecordDecl *RD,
                                  raw_ostream &OS,
                                  bool Simple) const {
  const ASTRecordLayout &Info = getASTRecordLayout(RD);

  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD))
    if (!Simple)
      return DumpCXXRecordLayout(OS, CXXRD, *this, CharUnits(), 0, 0,
                                 /*IncludeVirtualBases=*/true);

  OS << "Type: " << getTypeDeclType(RD).getAsString() << "\n";
  if (!Simple) {
    OS << "Record: ";
    RD->dump();
  }
  OS << "\nLayout: ";
  OS << "<ASTRecordLayout\n";
  OS << "  Size:" << toBits(Info.getSize()) << "\n";
  OS << "  DataSize:" << toBits(Info.getDataSize()) << "\n";
  OS << "  Alignment:" << toBits(Info.getAlignment()) << "\n";
  OS << "  FieldOffsets: [";
  for (unsigned i = 0, e = Info.getFieldCount(); i != e; ++i) {
    if (i) OS << ", ";
    OS << Info.getFieldOffset(i);
  }
  OS << "]>\n";
}
