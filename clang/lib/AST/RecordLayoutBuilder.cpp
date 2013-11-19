//=== RecordLayoutBuilder.cpp - Helper class for building record layouts ---==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/RecordLayout.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"

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

  // If we have empty structures inside a union, we can assign both
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

  /// UnfilledBitsInLastUnit - If the last field laid out was a bitfield,
  /// this contains the number of bits in the last unit that can be used for
  /// an adjacent bitfield if necessary.  The unit in question is usually
  /// a byte, but larger units are used if IsMsStruct.
  unsigned char UnfilledBitsInLastUnit;
  /// LastBitfieldTypeSize - If IsMsStruct, represents the size of the type
  /// of the previous field if it was a bitfield.
  unsigned char LastBitfieldTypeSize;

  /// MaxFieldAlignment - The maximum allowed field alignment. This is set by
  /// #pragma pack.
  CharUnits MaxFieldAlignment;

  /// DataSize - The data size of the record being laid out.
  uint64_t DataSize;

  CharUnits NonVirtualSize;
  CharUnits NonVirtualAlignment;

  /// PrimaryBase - the primary base class (if one exists) of the class
  /// we're laying out.
  const CXXRecordDecl *PrimaryBase;

  /// PrimaryBaseIsVirtual - Whether the primary base of the class we're laying
  /// out is virtual.
  bool PrimaryBaseIsVirtual;

  /// HasOwnVFPtr - Whether the class provides its own vtable/vftbl
  /// pointer, as opposed to inheriting one from a primary base class.
  bool HasOwnVFPtr;

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
      UnfilledBitsInLastUnit(0), LastBitfieldTypeSize(0),
      MaxFieldAlignment(CharUnits::Zero()), 
      DataSize(0), NonVirtualSize(CharUnits::Zero()), 
      NonVirtualAlignment(CharUnits::One()), 
      PrimaryBase(0), PrimaryBaseIsVirtual(false),
      HasOwnVFPtr(false),
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

  TargetCXXABI getCXXABI() const {
    return Context.getTargetInfo().getCXXABI();
  }

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

  /// LayoutVirtualBases - Lays out all the virtual bases.
  void LayoutVirtualBases(const CXXRecordDecl *RD,
                          const CXXRecordDecl *MostDerivedClass);

  /// LayoutVirtualBase - Lays out a single virtual base.
  void LayoutVirtualBase(const BaseSubobjectInfo *Base);

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

    if (Base->isDynamicClass()) {
      // We found it.
      PrimaryBase = Base;
      PrimaryBaseIsVirtual = false;
      return;
    }
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
  } else if (RD->isDynamicClass()) {
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

  // Now lay out the non-virtual bases.
  for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {

    // Ignore virtual bases.
    if (I->isVirtual())
      continue;

    const CXXRecordDecl *BaseDecl =
      cast<CXXRecordDecl>(I->getType()->castAs<RecordType>()->getDecl());

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

void RecordLayoutBuilder::LayoutVirtualBase(const BaseSubobjectInfo *Base) {
  assert(!Base->Derived && "Trying to lay out a primary virtual base!");
  
  // Layout the base.
  CharUnits Offset = LayoutBase(Base);

  // Add its base class offset.
  assert(!VBases.count(Base->Class) && "vbase offset already exists!");
  VBases.insert(std::make_pair(Base->Class, 
                       ASTRecordLayout::VBaseInfo(Offset, false)));

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
  
  CharUnits UnpackedBaseAlign = Layout.getNonVirtualAlign();
  CharUnits BaseAlign = (Packed) ? CharUnits::One() : UnpackedBaseAlign;
 
  // If we have an empty base class, try to place it at offset 0.
  if (Base->Class->isEmpty() &&
      (!HasExternalLayout || Offset == CharUnits::Zero()) &&
      EmptySubobjects->CanPlaceBaseAtOffset(Base, CharUnits::Zero())) {
    setSize(std::max(getSize(), Layout.getSize()));
    UpdateAlignment(BaseAlign, UnpackedBaseAlign);

    return CharUnits::Zero();
  }

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

  // Lay out the virtual bases and add the primary virtual base offsets.
  LayoutVirtualBases(RD, RD);

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
  for (RecordDecl::field_iterator Field = D->field_begin(),
       FieldEnd = D->field_end(); Field != FieldEnd; ++Field)
    LayoutField(*Field);
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
  UnfilledBitsInLastUnit = 0;
  LastBitfieldTypeSize = 0;

  uint64_t FieldOffset;
  uint64_t UnpaddedFieldOffset = getDataSizeInBits() - UnfilledBitsInLastUnit;

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
    UnfilledBitsInLastUnit = getDataSizeInBits() - NewSizeInBits;
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
  uint64_t FieldSize = D->getBitWidthValue(Context);
  std::pair<uint64_t, unsigned> FieldInfo = Context.getTypeInfo(D->getType());
  uint64_t TypeSize = FieldInfo.first;
  unsigned FieldAlign = FieldInfo.second;

  if (IsMsStruct) {
    // The field alignment for integer types in ms_struct structs is
    // always the size.
    FieldAlign = TypeSize;
    // Ignore zero-length bitfields after non-bitfields in ms_struct structs.
    if (!FieldSize && !LastBitfieldTypeSize)
      FieldAlign = 1;
    // If a bitfield is followed by a bitfield of a different size, don't
    // pack the bits together in ms_struct structs.
    if (LastBitfieldTypeSize != TypeSize) {
      UnfilledBitsInLastUnit = 0;
      LastBitfieldTypeSize = 0;
    }
  }

  uint64_t UnpaddedFieldOffset = getDataSizeInBits() - UnfilledBitsInLastUnit;
  uint64_t FieldOffset = IsUnion ? 0 : UnpaddedFieldOffset;

  bool ZeroLengthBitfield = false;
  if (!Context.getTargetInfo().useBitFieldTypeAlignment() &&
      Context.getTargetInfo().useZeroLengthBitfieldAlignment() &&
      FieldSize == 0) {
    // The alignment of a zero-length bitfield affects the alignment
    // of the next member.  The alignment is the max of the zero 
    // length bitfield's alignment and a target specific fixed value.
    ZeroLengthBitfield = true;
    unsigned ZeroLengthBitfieldBoundary =
      Context.getTargetInfo().getZeroLengthBitfieldBoundary();
    if (ZeroLengthBitfieldBoundary > FieldAlign)
      FieldAlign = ZeroLengthBitfieldBoundary;
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

  // ms_struct bitfields always have to start at a round alignment.
  if (IsMsStruct && !LastBitfieldTypeSize) {
    FieldOffset = llvm::RoundUpToAlignment(FieldOffset, FieldAlign);
    UnpackedFieldOffset = llvm::RoundUpToAlignment(UnpackedFieldOffset,
                                                   UnpackedFieldAlign);
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
  if (!D->getIdentifier() &&
      !Context.getTargetInfo().useZeroLengthBitfieldAlignment() &&
      !IsMsStruct)
    FieldAlign = UnpackedFieldAlign = 1;

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
    if (IsMsStruct && FieldSize) {
      // Under ms_struct, a bitfield always takes up space equal to the size
      // of the type.  We can't just change the alignment computation on the
      // other codepath because of the way this interacts with #pragma pack:
      // in a packed struct, we need to allocate misaligned space in the
      // struct to hold the bitfield.
      if (!UnfilledBitsInLastUnit) {
        setDataSize(FieldOffset + TypeSize);
        UnfilledBitsInLastUnit = TypeSize - FieldSize;
      } else if (UnfilledBitsInLastUnit < FieldSize) {
        setDataSize(getDataSizeInBits() + TypeSize);
        UnfilledBitsInLastUnit = TypeSize - FieldSize;
      } else {
        UnfilledBitsInLastUnit -= FieldSize;
      }
      LastBitfieldTypeSize = TypeSize;
    } else {
      uint64_t NewSizeInBits = FieldOffset + FieldSize;
      uint64_t BitfieldAlignment = Context.getTargetInfo().getCharAlign();
      setDataSize(llvm::RoundUpToAlignment(NewSizeInBits, BitfieldAlignment));
      UnfilledBitsInLastUnit = getDataSizeInBits() - NewSizeInBits;
      LastBitfieldTypeSize = 0;
    }
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

  uint64_t UnpaddedFieldOffset = getDataSizeInBits() - UnfilledBitsInLastUnit;

  // Reset the unfilled bits.
  UnfilledBitsInLastUnit = 0;
  LastBitfieldTypeSize = 0;

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
  uint64_t UnpaddedSize = getSizeInBits() - UnfilledBitsInLastUnit;
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

static const CXXMethodDecl *computeKeyFunction(ASTContext &Context,
                                               const CXXRecordDecl *RD) {
  // If a class isn't polymorphic it doesn't have a key function.
  if (!RD->isPolymorphic())
    return 0;

  // A class that is not externally visible doesn't have a key function. (Or
  // at least, there's no point to assigning a key function to such a class;
  // this doesn't affect the ABI.)
  if (!RD->isExternallyVisible())
    return 0;

  // Template instantiations don't have key functions,see Itanium C++ ABI 5.2.6.
  // Same behavior as GCC.
  TemplateSpecializationKind TSK = RD->getTemplateSpecializationKind();
  if (TSK == TSK_ImplicitInstantiation ||
      TSK == TSK_ExplicitInstantiationDefinition)
    return 0;

  bool allowInlineFunctions =
    Context.getTargetInfo().getCXXABI().canKeyFunctionBeInline();

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

    // In certain ABIs, ignore functions with out-of-line inline definitions.
    if (!allowInlineFunctions) {
      const FunctionDecl *Def;
      if (MD->hasBody(Def) && Def->isInlineSpecified())
        continue;
    }

    // We found it.
    return MD;
  }

  return 0;
}

DiagnosticBuilder
RecordLayoutBuilder::Diag(SourceLocation Loc, unsigned DiagID) {
  return Context.getDiagnostics().Report(Loc, DiagID);
}

/// Does the target C++ ABI require us to skip over the tail-padding
/// of the given class (considering it as a base class) when allocating
/// objects?
static bool mustSkipTailPadding(TargetCXXABI ABI, const CXXRecordDecl *RD) {
  switch (ABI.getTailPaddingUseRules()) {
  case TargetCXXABI::AlwaysUseTailPadding:
    return false;

  case TargetCXXABI::UseTailPaddingUnlessPOD03:
    // FIXME: To the extent that this is meant to cover the Itanium ABI
    // rules, we should implement the restrictions about over-sized
    // bitfields:
    //
    // http://mentorembedded.github.com/cxx-abi/abi.html#POD :
    //   In general, a type is considered a POD for the purposes of
    //   layout if it is a POD type (in the sense of ISO C++
    //   [basic.types]). However, a POD-struct or POD-union (in the
    //   sense of ISO C++ [class]) with a bitfield member whose
    //   declared width is wider than the declared type of the
    //   bitfield is not a POD for the purpose of layout.  Similarly,
    //   an array type is not a POD for the purpose of layout if the
    //   element type of the array is not a POD for the purpose of
    //   layout.
    //
    //   Where references to the ISO C++ are made in this paragraph,
    //   the Technical Corrigendum 1 version of the standard is
    //   intended.
    return RD->isPOD();

  case TargetCXXABI::UseTailPaddingUnlessPOD11:
    // This is equivalent to RD->getTypeForDecl().isCXX11PODType(),
    // but with a lot of abstraction penalty stripped off.  This does
    // assume that these properties are set correctly even in C++98
    // mode; fortunately, that is true because we want to assign
    // consistently semantics to the type-traits intrinsics (or at
    // least as many of them as possible).
    return RD->isTrivial() && RD->isStandardLayout();
  }

  llvm_unreachable("bad tail-padding use kind");
}

static bool isMsLayout(const RecordDecl* D) {
  return D->getASTContext().getTargetInfo().getCXXABI().isMicrosoft();
}

// This section contains an implementation of struct layout that is, up to the
// included tests, compatible with cl.exe (2012).  The layout produced is
// significantly different than those produced by the Itanium ABI.  Here we note
// the most important differences.
//
// * The alignment of bitfields in unions is ignored when computing the
//   alignment of the union.
// * The existance of zero-width bitfield that occurs after anything other than
//   a non-zero length bitfield is ignored.
// * The Itanium equivalent vtable pointers are split into a vfptr (virtual
//   function pointer) and a vbptr (virtual base pointer).  They can each be
//   shared with a, non-virtual bases. These bases need not be the same.  vfptrs
//   always occur at offset 0.  vbptrs can occur at an
//   arbitrary offset and are placed after non-virtual bases but before fields.
// * Virtual bases sometimes require a 'vtordisp' field that is laid out before
//   the virtual base and is used in conjunction with virtual overrides during
//   construction and destruction.
// * vfptrs are allocated in a block of memory equal to the alignment of the
//   fields and non-virtual bases at offset 0 in 32 bit mode and in a pointer
//   sized block of memory in 64 bit mode.
// * vbptrs are allocated in a block of memory equal to the alignment of the
//   fields and non-virtual bases.  This block is at a potentially unaligned
//   offset.  If the allocation slot is unaligned and the alignment is less than
//   or equal to the pointer size, additional space is allocated so that the
//   pointer can be aligned properly.  This causes very strange effects on the
//   placement of objects after the allocated block. (see the code).
// * vtordisps are allocated in a block of memory with size and alignment equal
//   to the alignment of the completed structure (before applying __declspec(
//   align())).  The vtordisp always occur at the end of the allocation block,
//   immediately prior to the virtual base.
// * The last zero sized non-virtual base is allocated after the placement of
//   vbptr if one exists and can be placed at the end of the struct, potentially
//   aliasing either the first member or another struct allocated after this
//   one.
// * The last zero size virtual base may be placed at the end of the struct.
//   and can potentially alias a zero sized type in the next struct.
// * If the last field is a non-zero length bitfield and we have any virtual
//   bases then some extra padding is added before the virtual bases for no
//   obvious reason.
// * When laying out empty non-virtual bases, an extra byte of padding is added
//   if the non-virtual base before the empty non-virtual base has a vbptr.


namespace {
struct MicrosoftRecordLayoutBuilder {
  typedef llvm::DenseMap<const CXXRecordDecl *, CharUnits> BaseOffsetsMapTy;
  MicrosoftRecordLayoutBuilder(const ASTContext &Context) : Context(Context) {}
private:
  MicrosoftRecordLayoutBuilder(const MicrosoftRecordLayoutBuilder &)
  LLVM_DELETED_FUNCTION;
  void operator=(const MicrosoftRecordLayoutBuilder &) LLVM_DELETED_FUNCTION;
public:

  void layout(const RecordDecl *RD);
  void cxxLayout(const CXXRecordDecl *RD);
  /// \brief Initializes size and alignment and honors some flags.
  void initializeLayout(const RecordDecl *RD);
  /// \brief Initialized C++ layout, compute alignment and virtual alignment and
  /// existance of vfptrs and vbptrs.  Alignment is needed before the vfptr is
  /// laid out.
  void initializeCXXLayout(const CXXRecordDecl *RD);
  void layoutVFPtr(const CXXRecordDecl *RD);
  void layoutNonVirtualBases(const CXXRecordDecl *RD);
  void layoutNonVirtualBase(const CXXRecordDecl *RD);
  void layoutVBPtr(const CXXRecordDecl *RD);
  /// \brief Lays out the fields of the record.  Also rounds size up to
  /// alignment.
  void layoutFields(const RecordDecl *RD);
  void layoutField(const FieldDecl *FD);
  void layoutBitField(const FieldDecl *FD);
  /// \brief Lays out a single zero-width bit-field in the record and handles
  /// special cases associated with zero-width bit-fields.
  void layoutZeroWidthBitField(const FieldDecl *FD);
  void layoutVirtualBases(const CXXRecordDecl *RD);
  void layoutVirtualBase(const CXXRecordDecl *RD, bool HasVtordisp);
  /// \brief Flushes the lazy virtual base and conditionally rounds up to
  /// alignment.
  void finalizeCXXLayout(const CXXRecordDecl *RD);
  void honorDeclspecAlign(const RecordDecl *RD);

  /// \brief Updates the alignment of the type.  This function doesn't take any
  /// properties (such as packedness) into account.  getAdjustedFieldInfo()
  /// adjustes for packedness.
  void updateAlignment(CharUnits NewAlignment) {
    Alignment = std::max(Alignment, NewAlignment);
  }
  /// \brief Gets the size and alignment taking attributes into account.
  std::pair<CharUnits, CharUnits> getAdjustedFieldInfo(const FieldDecl *FD);
  /// \brief Places a field at offset 0.
  void placeFieldAtZero() { FieldOffsets.push_back(0); }
  /// \brief Places a field at an offset in CharUnits.
  void placeFieldAtOffset(CharUnits FieldOffset) {
    FieldOffsets.push_back(Context.toBits(FieldOffset));
  }
  /// \brief Places a bitfield at a bit offset.
  void placeFieldAtBitOffset(uint64_t FieldOffset) {
    FieldOffsets.push_back(FieldOffset);
  }
  /// \brief Compute the set of virtual bases for which vtordisps are required.
  llvm::SmallPtrSet<const CXXRecordDecl *, 2>
  computeVtorDispSet(const CXXRecordDecl *RD);

  const ASTContext &Context;
  /// \brief The size of the record being laid out.
  CharUnits Size;
  /// \brief The current alignment of the record layout.
  CharUnits Alignment;
  /// \brief The collection of field offsets.
  SmallVector<uint64_t, 16> FieldOffsets;
  /// \brief The maximum allowed field alignment. This is set by #pragma pack.
  CharUnits MaxFieldAlignment;
  /// \brief Alignment does not occur for virtual bases unless something
  /// forces it to by explicitly using __declspec(align())
  bool AlignAfterVBases : 1;
  bool IsUnion : 1;
  /// \brief True if the last field laid out was a bitfield and was not 0
  /// width.
  bool LastFieldIsNonZeroWidthBitfield : 1;
  /// \brief The size of the allocation of the currently active bitfield.
  /// This value isn't meaningful unless LastFieldIsNonZeroWidthBitfield
  /// is true.
  CharUnits CurrentBitfieldSize;
  /// \brief The number of remaining bits in our last bitfield allocation.
  /// This value isn't meaningful unless LastFieldIsNonZeroWidthBitfield is
  /// true.
  unsigned RemainingBitsInField;

  /// \brief The data alignment of the record layout.
  CharUnits DataSize;
  /// \brief The alignment of the non-virtual portion of the record layout
  /// without the impact of the virtual pointers.
  /// Only used for C++ layouts.
  CharUnits BasesAndFieldsAlignment;
  /// \brief The alignment of the non-virtual portion of the record layout
  /// Only used for C++ layouts.
  CharUnits NonVirtualAlignment;
  /// \brief The additional alignment imposed by the virtual bases.
  CharUnits VirtualAlignment;
  /// \brief The primary base class (if one exists).
  const CXXRecordDecl *PrimaryBase;
  /// \brief The class we share our vb-pointer with.
  const CXXRecordDecl *SharedVBPtrBase;
  /// \brief True if the class has a vftable pointer that can be extended
  /// by this class or classes derived from it.  Such a vfptr will always occur
  /// at offset 0.
  bool HasExtendableVFPtr : 1;
  /// \brief True if the class has a (not necessarily its own) vbtable pointer.
  bool HasVBPtr : 1;
  /// \brief Offset to the virtual base table pointer (if one exists).
  CharUnits VBPtrOffset;
  /// \brief Base classes and their offsets in the record.
  BaseOffsetsMapTy Bases;
  /// \brief virtual base classes and their offsets in the record.
  ASTRecordLayout::VBaseOffsetsMapTy VBases;
  /// \brief The size of a pointer.
  CharUnits PointerSize;
  /// \brief The alignment of a pointer.
  CharUnits PointerAlignment;
  /// \brief Holds an empty base we haven't yet laid out.
  const CXXRecordDecl *LazyEmptyBase;
  /// \brief Lets us know if the last base we laid out was empty.  Only used
  /// when adjusting the placement of a last zero-sized base in 64 bit mode.
  bool LastBaseWasEmpty;
  /// \brief Lets us know if we're in 64-bit mode
  bool Is64BitMode;
  /// \brief True if the last non-virtual base has a vbptr.
  bool LastNonVirtualBaseHasVBPtr;
};
} // namespace

std::pair<CharUnits, CharUnits>
MicrosoftRecordLayoutBuilder::getAdjustedFieldInfo(const FieldDecl *FD) {
  std::pair<CharUnits, CharUnits> FieldInfo =
      Context.getTypeInfoInChars(FD->getType());

  // If we're not on win32 and using ms_struct the field alignment will be wrong
  // for 64 bit types, so we fix that here.
  if (FD->getASTContext().getTargetInfo().getTriple().getOS() !=
      llvm::Triple::Win32) {
    QualType T = Context.getBaseElementType(FD->getType());
    if (const BuiltinType *BTy = T->getAs<BuiltinType>()) {
      CharUnits TypeSize = Context.getTypeSizeInChars(BTy);
      if (TypeSize > FieldInfo.second)
        FieldInfo.second = TypeSize;
    }
  }

  // Respect packed attribute.
  if (FD->hasAttr<PackedAttr>())
    FieldInfo.second = CharUnits::One();
  // Respect pack pragma.
  else if (!MaxFieldAlignment.isZero())
    FieldInfo.second = std::min(FieldInfo.second, MaxFieldAlignment);
  // Respect alignment attributes.
  if (unsigned fieldAlign = FD->getMaxAlignment()) {
    CharUnits FieldAlign = Context.toCharUnitsFromBits(fieldAlign);
    AlignAfterVBases = true;
    FieldInfo.second = std::max(FieldInfo.second, FieldAlign);
  }
  return FieldInfo;
}

void MicrosoftRecordLayoutBuilder::initializeLayout(const RecordDecl *RD) {
  IsUnion = RD->isUnion();
  Is64BitMode = Context.getTargetInfo().getPointerWidth(0) == 64;

  Size = CharUnits::Zero();
  Alignment = CharUnits::One();
  AlignAfterVBases = false;

  // Compute the maximum field alignment.
  MaxFieldAlignment = CharUnits::Zero();
  // Honor the default struct packing maximum alignment flag.
  if (unsigned DefaultMaxFieldAlignment = Context.getLangOpts().PackStruct)
    MaxFieldAlignment = CharUnits::fromQuantity(DefaultMaxFieldAlignment);
  // Honor the packing attribute.
  if (const MaxFieldAlignmentAttr *MFAA = RD->getAttr<MaxFieldAlignmentAttr>())
    MaxFieldAlignment = Context.toCharUnitsFromBits(MFAA->getAlignment());
  // Packed attribute forces max field alignment to be 1.
  if (RD->hasAttr<PackedAttr>())
    MaxFieldAlignment = CharUnits::One();
}

void MicrosoftRecordLayoutBuilder::layout(const RecordDecl *RD) {
  initializeLayout(RD);
  layoutFields(RD);
  honorDeclspecAlign(RD);
}

void MicrosoftRecordLayoutBuilder::cxxLayout(const CXXRecordDecl *RD) {
  initializeLayout(RD);
  initializeCXXLayout(RD);
  layoutVFPtr(RD);
  layoutNonVirtualBases(RD);
  layoutVBPtr(RD);
  layoutFields(RD);
  DataSize = Size;
  NonVirtualAlignment = Alignment;
  layoutVirtualBases(RD);
  finalizeCXXLayout(RD);
  honorDeclspecAlign(RD);
}

void
MicrosoftRecordLayoutBuilder::initializeCXXLayout(const CXXRecordDecl *RD) {
  // Calculate pointer size and alignment.
  PointerSize =
      Context.toCharUnitsFromBits(Context.getTargetInfo().getPointerWidth(0));
  PointerAlignment = PointerSize;
  if (!MaxFieldAlignment.isZero())
    PointerAlignment = std::min(PointerAlignment, MaxFieldAlignment);

  // Initialize information about the bases.
  HasVBPtr = false;
  HasExtendableVFPtr = false;
  SharedVBPtrBase = 0;
  PrimaryBase = 0;
  VirtualAlignment = CharUnits::One();
  AlignAfterVBases = Is64BitMode;

  // If the record has a dynamic base class, attempt to choose a primary base
  // class. It is the first (in direct base class order) non-virtual dynamic
  // base class, if one exists.
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
                                                e = RD->bases_end();
       i != e; ++i) {
    const CXXRecordDecl *BaseDecl =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(BaseDecl);
    // Handle forced alignment.
    if (Layout.getAlignAfterVBases())
      AlignAfterVBases = true;
    // Handle virtual bases.
    if (i->isVirtual()) {
      VirtualAlignment = std::max(VirtualAlignment, Layout.getAlignment());
      HasVBPtr = true;
      continue;
    }
    // We located a primary base class!
    if (!PrimaryBase && Layout.hasExtendableVFPtr()) {
      PrimaryBase = BaseDecl;
      HasExtendableVFPtr = true;
    }
    // We located a base to share a VBPtr with!
    if (!SharedVBPtrBase && Layout.hasVBPtr()) {
      SharedVBPtrBase = BaseDecl;
      HasVBPtr = true;
    }
    updateAlignment(Layout.getAlignment());
  }

  // Use LayoutFields to compute the alignment of the fields.  The layout
  // is discarded.  This is the simplest way to get all of the bit-field
  // behavior correct and is not actually very expensive.
  layoutFields(RD);
  Size = CharUnits::Zero();
  BasesAndFieldsAlignment = Alignment;
  FieldOffsets.clear();
}

void MicrosoftRecordLayoutBuilder::layoutVFPtr(const CXXRecordDecl *RD) {
  // If we have a primary base then our VFPtr was already laid out
  if (PrimaryBase)
    return;

  // Look at all of our methods to determine if we need a VFPtr.  We need a
  // vfptr if we define a new virtual function.
  if (!HasExtendableVFPtr && RD->isDynamicClass())
    for (CXXRecordDecl::method_iterator i = RD->method_begin(),
                                        e = RD->method_end();
         !HasExtendableVFPtr && i != e; ++i)
      HasExtendableVFPtr = i->isVirtual() && i->size_overridden_methods() == 0;
  if (!HasExtendableVFPtr)
    return;

  // MSVC 32 (but not 64) potentially over-aligns the vf-table pointer by giving
  // it the max alignment of all the non-virtual data in the class.  The
  // resulting layout is essentially { vftbl, { nvdata } }.  This is completely
  // unnecessary, but we're not here to pass judgment.
  updateAlignment(PointerAlignment);
  if (Is64BitMode)
    Size = Size.RoundUpToAlignment(PointerAlignment) + PointerSize;
  else
    Size = Size.RoundUpToAlignment(PointerAlignment) + Alignment;
}

void
MicrosoftRecordLayoutBuilder::layoutNonVirtualBases(const CXXRecordDecl *RD) {
  LazyEmptyBase = 0;
  LastBaseWasEmpty = false;
  LastNonVirtualBaseHasVBPtr = false;

  // Lay out the primary base first.
  if (PrimaryBase)
    layoutNonVirtualBase(PrimaryBase);

  // Iterate through the bases and lay out the non-virtual ones.
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
                                                e = RD->bases_end();
       i != e; ++i) {
    if (i->isVirtual())
      continue;
    const CXXRecordDecl *BaseDecl =
        cast<CXXRecordDecl>(i->getType()->castAs<RecordType>()->getDecl());
    if (BaseDecl != PrimaryBase)
      layoutNonVirtualBase(BaseDecl);
  }
}

void
MicrosoftRecordLayoutBuilder::layoutNonVirtualBase(const CXXRecordDecl *RD) {
  const ASTRecordLayout *Layout = RD ? &Context.getASTRecordLayout(RD) : 0;

  // If we have a lazy empty base we haven't laid out yet, do that now.
  if (LazyEmptyBase) {
    const ASTRecordLayout &LazyLayout =
        Context.getASTRecordLayout(LazyEmptyBase);
    Size = Size.RoundUpToAlignment(LazyLayout.getAlignment());
    // If the last non-virtual base has a vbptr we add a byte of padding for no
    // obvious reason.
    if (LastNonVirtualBaseHasVBPtr)
      Size++;
    Bases.insert(std::make_pair(LazyEmptyBase, Size));
    // Empty bases only consume space when followed by another empty base.
    if (RD && Layout->getNonVirtualSize().isZero()) {
      LastBaseWasEmpty = true;
      Size++;
    }
    LazyEmptyBase = 0;
    LastNonVirtualBaseHasVBPtr = false;
  }

  // RD is null when flushing the final lazy base.
  if (!RD)
    return;

  if (Layout->getNonVirtualSize().isZero()) {
    LazyEmptyBase = RD;
    return;
  }

  // Insert the base here.
  CharUnits BaseOffset = Size.RoundUpToAlignment(Layout->getAlignment());
  Bases.insert(std::make_pair(RD, BaseOffset));
  Size = BaseOffset + Layout->getDataSize();
  // Note: we don't update alignment here because it was accounted
  // for during initalization.
  LastBaseWasEmpty = false;
  LastNonVirtualBaseHasVBPtr = Layout->hasVBPtr();
}

void MicrosoftRecordLayoutBuilder::layoutVBPtr(const CXXRecordDecl *RD) {
  if (!HasVBPtr)
    VBPtrOffset = CharUnits::fromQuantity(-1);
  else if (SharedVBPtrBase) {
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(SharedVBPtrBase);
    VBPtrOffset = Bases[SharedVBPtrBase] + Layout.getVBPtrOffset();
  } else {
    VBPtrOffset = Size.RoundUpToAlignment(PointerAlignment);
    CharUnits OldSize = Size;
    Size = VBPtrOffset + PointerSize;
    if (BasesAndFieldsAlignment <= PointerAlignment) {
      // Handle strange padding rules for the lazily placed base.  I have no
      // explanation for why the last virtual base is padded in such an odd way.
      // Two things to note about this padding are that the rules are different
      // if the alignment of the bases+fields is <= to the alignemnt of a
      // pointer and that the rule in 64-bit mode behaves differently depending
      // on if the second to last base was also zero sized.
      Size += OldSize % BasesAndFieldsAlignment.getQuantity();
    } else {
      if (Is64BitMode)
        Size += LastBaseWasEmpty ? CharUnits::One() : CharUnits::Zero();
      else
        Size = OldSize + BasesAndFieldsAlignment;
    }
    updateAlignment(PointerAlignment);
  }

  // Flush the lazy empty base.
  layoutNonVirtualBase(0);
}

void MicrosoftRecordLayoutBuilder::layoutFields(const RecordDecl *RD) {
  LastFieldIsNonZeroWidthBitfield = false;
  for (RecordDecl::field_iterator Field = RD->field_begin(),
                                  FieldEnd = RD->field_end();
       Field != FieldEnd; ++Field)
    layoutField(*Field);
  Size = Size.RoundUpToAlignment(Alignment);
}

void MicrosoftRecordLayoutBuilder::layoutField(const FieldDecl *FD) {
  if (FD->isBitField()) {
    layoutBitField(FD);
    return;
  }
  LastFieldIsNonZeroWidthBitfield = false;

  std::pair<CharUnits, CharUnits> FieldInfo = getAdjustedFieldInfo(FD);
  CharUnits FieldSize = FieldInfo.first;
  CharUnits FieldAlign = FieldInfo.second;

  updateAlignment(FieldAlign);
  if (IsUnion) {
    placeFieldAtZero();
    Size = std::max(Size, FieldSize);
  } else {
    // Round up the current record size to the field's alignment boundary.
    CharUnits FieldOffset = Size.RoundUpToAlignment(FieldAlign);
    placeFieldAtOffset(FieldOffset);
    Size = FieldOffset + FieldSize;
  }
}

void MicrosoftRecordLayoutBuilder::layoutBitField(const FieldDecl *FD) {
  unsigned Width = FD->getBitWidthValue(Context);
  if (Width == 0) {
    layoutZeroWidthBitField(FD);
    return;
  }

  std::pair<CharUnits, CharUnits> FieldInfo = getAdjustedFieldInfo(FD);
  CharUnits FieldSize = FieldInfo.first;
  CharUnits FieldAlign = FieldInfo.second;

  // Clamp the bitfield to a containable size for the sake of being able
  // to lay them out.  Sema will throw an error.
  if (Width > Context.toBits(FieldSize))
    Width = Context.toBits(FieldSize);

  // Check to see if this bitfield fits into an existing allocation.  Note:
  // MSVC refuses to pack bitfields of formal types with different sizes
  // into the same allocation.
  if (!IsUnion && LastFieldIsNonZeroWidthBitfield &&
      CurrentBitfieldSize == FieldSize && Width <= RemainingBitsInField) {
    placeFieldAtBitOffset(Context.toBits(Size) - RemainingBitsInField);
    RemainingBitsInField -= Width;
    return;
  }

  LastFieldIsNonZeroWidthBitfield = true;
  CurrentBitfieldSize = FieldSize;
  if (IsUnion) {
    placeFieldAtZero();
    Size = std::max(Size, FieldSize);
    // TODO: Add a Sema warning that MS ignores bitfield alignment in unions.
  } else {
    // Allocate a new block of memory and place the bitfield in it.
    CharUnits FieldOffset = Size.RoundUpToAlignment(FieldAlign);
    placeFieldAtOffset(FieldOffset);
    Size = FieldOffset + FieldSize;
    updateAlignment(FieldAlign);
    RemainingBitsInField = Context.toBits(FieldSize) - Width;
  }
}

void
MicrosoftRecordLayoutBuilder::layoutZeroWidthBitField(const FieldDecl *FD) {
  // Zero-width bitfields are ignored unless they follow a non-zero-width
  // bitfield.
  std::pair<CharUnits, CharUnits> FieldInfo = getAdjustedFieldInfo(FD);
  CharUnits FieldSize = FieldInfo.first;
  CharUnits FieldAlign = FieldInfo.second;

  if (!LastFieldIsNonZeroWidthBitfield) {
    placeFieldAtOffset(IsUnion ? CharUnits::Zero() : Size);
    // TODO: Add a Sema warning that MS ignores alignment for zero
    // sized bitfields that occur after zero-size bitfields or non bitfields.
    return;
  }

  LastFieldIsNonZeroWidthBitfield = false;
  if (IsUnion) {
    placeFieldAtZero();
    Size = std::max(Size, FieldSize);
  } else {
    // Round up the current record size to the field's alignment boundary.
    CharUnits FieldOffset = Size.RoundUpToAlignment(FieldAlign);
    placeFieldAtOffset(FieldOffset);
    Size = FieldOffset;
    updateAlignment(FieldAlign);
  }
}

void MicrosoftRecordLayoutBuilder::layoutVirtualBases(const CXXRecordDecl *RD) {
  if (!HasVBPtr)
    return;

  updateAlignment(VirtualAlignment);

  // Zero-sized v-bases obey the alignment attribute so apply it here.  The
  // alignment attribute is normally accounted for in FinalizeLayout.
  if (unsigned MaxAlign = RD->getMaxAlignment())
    updateAlignment(Context.toCharUnitsFromBits(MaxAlign));

  llvm::SmallPtrSet<const CXXRecordDecl *, 2> HasVtordisp =
      computeVtorDispSet(RD);

  // If the last field we laid out was a non-zero length bitfield then add some
  // extra padding for no obvious reason.
  if (LastFieldIsNonZeroWidthBitfield)
    Size += CurrentBitfieldSize;

  // Iterate through the virtual bases and lay them out.
  for (CXXRecordDecl::base_class_const_iterator i = RD->vbases_begin(),
                                                e = RD->vbases_end();
       i != e; ++i) {
    const CXXRecordDecl *BaseDecl =
        cast<CXXRecordDecl>(i->getType()->castAs<RecordType>()->getDecl());
    layoutVirtualBase(BaseDecl, HasVtordisp.count(BaseDecl));
  }
}

void MicrosoftRecordLayoutBuilder::layoutVirtualBase(const CXXRecordDecl *RD,
                                                     bool HasVtordisp) {
  if (LazyEmptyBase) {
    const ASTRecordLayout &LazyLayout =
        Context.getASTRecordLayout(LazyEmptyBase);
    Size = Size.RoundUpToAlignment(LazyLayout.getAlignment());
    VBases.insert(
        std::make_pair(LazyEmptyBase, ASTRecordLayout::VBaseInfo(Size, false)));
    // Empty bases only consume space when followed by another empty base.
    // The space consumed is in an Alignment sized/aligned block and the v-base
    // is placed at its alignment offset into the chunk, unless its alignment
    // is less than 4 bytes, at which it is placed at 4 byte offset in the
    // chunk.  We have no idea why.
    if (RD && Context.getASTRecordLayout(RD).getNonVirtualSize().isZero())
      Size = Size.RoundUpToAlignment(Alignment) + CharUnits::fromQuantity(4);
    LazyEmptyBase = 0;
  }

  // RD is null when flushing the final lazy virtual base.
  if (!RD)
    return;

  const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
  if (Layout.getNonVirtualSize().isZero() && !HasVtordisp) {
    LazyEmptyBase = RD;
    return;
  }

  CharUnits BaseNVSize = Layout.getNonVirtualSize();
  CharUnits BaseAlign = Layout.getAlignment();

  // vtordisps are always 4 bytes (even in 64-bit mode)
  if (HasVtordisp)
    Size = Size.RoundUpToAlignment(Alignment) + CharUnits::fromQuantity(4);
  Size = Size.RoundUpToAlignment(BaseAlign);

  // Insert the base here.
  CharUnits BaseOffset = Size.RoundUpToAlignment(BaseAlign);
  VBases.insert(
      std::make_pair(RD, ASTRecordLayout::VBaseInfo(BaseOffset, HasVtordisp)));
  Size = BaseOffset + BaseNVSize;
  // Note: we don't update alignment here because it was accounted for in
  // InitializeLayout.
}

void MicrosoftRecordLayoutBuilder::finalizeCXXLayout(const CXXRecordDecl *RD) {
  // Flush the lazy virtual base.
  layoutVirtualBase(0, false);

  if (RD->vbases_begin() == RD->vbases_end() || AlignAfterVBases)
    Size = Size.RoundUpToAlignment(Alignment);

  if (Size.isZero())
    Size = Alignment;
}

void MicrosoftRecordLayoutBuilder::honorDeclspecAlign(const RecordDecl *RD) {
  if (unsigned MaxAlign = RD->getMaxAlignment()) {
    AlignAfterVBases = true;
    updateAlignment(Context.toCharUnitsFromBits(MaxAlign));
    Size = Size.RoundUpToAlignment(Alignment);
  }
}

static bool
RequiresVtordisp(const llvm::SmallPtrSet<const CXXRecordDecl *, 2> &HasVtordisp,
                 const CXXRecordDecl *RD) {
  if (HasVtordisp.count(RD))
    return true;
  // If any of a virtual bases non-virtual bases (recursively) requires a
  // vtordisp than so does this virtual base.
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
                                                e = RD->bases_end();
       i != e; ++i)
    if (!i->isVirtual() &&
        RequiresVtordisp(
            HasVtordisp,
            cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl())))
      return true;
  return false;
}

llvm::SmallPtrSet<const CXXRecordDecl *, 2>
MicrosoftRecordLayoutBuilder::computeVtorDispSet(const CXXRecordDecl *RD) {
  llvm::SmallPtrSet<const CXXRecordDecl *, 2> HasVtordisp;

  // If any of our bases need a vtordisp for this type, so do we.  Check our
  // direct bases for vtordisp requirements.
  for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
                                                e = RD->bases_end();
       i != e; ++i) {
    const CXXRecordDecl *BaseDecl =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(BaseDecl);
    for (ASTRecordLayout::VBaseOffsetsMapTy::const_iterator
             bi = Layout.getVBaseOffsetsMap().begin(),
             be = Layout.getVBaseOffsetsMap().end();
         bi != be; ++bi)
      if (bi->second.hasVtorDisp())
        HasVtordisp.insert(bi->first);
  }

  // If we define a constructor or destructor and override a function that is
  // defined in a virtual base's vtable, that virtual bases need a vtordisp.
  // Here we collect a list of classes with vtables for which our virtual bases
  // actually live.  The virtual bases with this property will require
  // vtordisps.  In addition, virtual bases that contain non-virtual bases that
  // define functions we override also require vtordisps, this case is checked
  // explicitly below.
  if (RD->hasUserDeclaredConstructor() || RD->hasUserDeclaredDestructor()) {
    llvm::SmallPtrSet<const CXXMethodDecl *, 8> Work;
    // Seed the working set with our non-destructor virtual methods.
    for (CXXRecordDecl::method_iterator i = RD->method_begin(),
                                        e = RD->method_end();
         i != e; ++i)
      if ((*i)->isVirtual() && !isa<CXXDestructorDecl>(*i))
        Work.insert(*i);
    while (!Work.empty()) {
      const CXXMethodDecl *MD = *Work.begin();
      CXXMethodDecl::method_iterator i = MD->begin_overridden_methods(),
                                     e = MD->end_overridden_methods();
      if (i == e)
        // If a virtual method has no-overrides it lives in its parent's vtable.
        HasVtordisp.insert(MD->getParent());
      else
        Work.insert(i, e);
      // We've finished processing this element, remove it from the working set.
      Work.erase(MD);
    }
  }

  // Re-check all of our vbases for vtordisp requirements (in case their
  // non-virtual bases have vtordisp requirements).
  for (CXXRecordDecl::base_class_const_iterator i = RD->vbases_begin(),
                                                e = RD->vbases_end();
       i != e; ++i) {
    const CXXRecordDecl *BaseDecl =  i->getType()->getAsCXXRecordDecl();
    if (!HasVtordisp.count(BaseDecl) && RequiresVtordisp(HasVtordisp, BaseDecl))
      HasVtordisp.insert(BaseDecl);
  }

  return HasVtordisp;
}

/// \brief Get or compute information about the layout of the specified record
/// (struct/union/class), which indicates its size and field position
/// information.
const ASTRecordLayout *
ASTContext::BuildMicrosoftASTRecordLayout(const RecordDecl *D) const {
  MicrosoftRecordLayoutBuilder Builder(*this);
  if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D)) {
    Builder.cxxLayout(RD);
    return new (*this) ASTRecordLayout(
        *this, Builder.Size, Builder.Alignment,
        Builder.HasExtendableVFPtr && !Builder.PrimaryBase,
        Builder.HasExtendableVFPtr,
        Builder.VBPtrOffset, Builder.DataSize, Builder.FieldOffsets.data(),
        Builder.FieldOffsets.size(), Builder.DataSize,
        Builder.NonVirtualAlignment, CharUnits::Zero(), Builder.PrimaryBase,
        false, Builder.SharedVBPtrBase, Builder.AlignAfterVBases, Builder.Bases,
        Builder.VBases);
  } else {
    Builder.layout(D);
    return new (*this) ASTRecordLayout(
        *this, Builder.Size, Builder.Alignment, Builder.Size,
        Builder.FieldOffsets.data(), Builder.FieldOffsets.size());
  }
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
  assert(!D->isInvalidDecl() && "Cannot get layout of invalid decl!");
  assert(D->isCompleteDefinition() && "Cannot layout type before complete!");

  // Look up this layout, if already laid out, return what we have.
  // Note that we can't save a reference to the entry because this function
  // is recursive.
  const ASTRecordLayout *Entry = ASTRecordLayouts[D];
  if (Entry) return *Entry;

  const ASTRecordLayout *NewEntry = 0;

  if (isMsLayout(D) && !D->getASTContext().getExternalSource()) {
    NewEntry = BuildMicrosoftASTRecordLayout(D);
  } else if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D)) {
    EmptySubobjectMap EmptySubobjects(*this, RD);
    RecordLayoutBuilder Builder(*this, &EmptySubobjects);
    Builder.Layout(RD);

    // In certain situations, we are allowed to lay out objects in the
    // tail-padding of base classes.  This is ABI-dependent.
    // FIXME: this should be stored in the record layout.
    bool skipTailPadding =
      mustSkipTailPadding(getTargetInfo().getCXXABI(), cast<CXXRecordDecl>(D));

    // FIXME: This should be done in FinalizeLayout.
    CharUnits DataSize =
      skipTailPadding ? Builder.getSize() : Builder.getDataSize();
    CharUnits NonVirtualSize = 
      skipTailPadding ? DataSize : Builder.NonVirtualSize;
    NewEntry =
      new (*this) ASTRecordLayout(*this, Builder.getSize(), 
                                  Builder.Alignment,
                                  Builder.HasOwnVFPtr,
                                  RD->isDynamicClass(),
                                  CharUnits::fromQuantity(-1),
                                  DataSize, 
                                  Builder.FieldOffsets.data(),
                                  Builder.FieldOffsets.size(),
                                  NonVirtualSize,
                                  Builder.NonVirtualAlignment,
                                  EmptySubobjects.SizeOfLargestEmptySubobject,
                                  Builder.PrimaryBase,
                                  Builder.PrimaryBaseIsVirtual,
                                  0, true,
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
    llvm::outs() << "\n*** Dumping AST Record Layout\n";
    DumpRecordLayout(D, llvm::outs(), getLangOpts().DumpRecordLayoutsSimple);
  }

  return *NewEntry;
}

const CXXMethodDecl *ASTContext::getCurrentKeyFunction(const CXXRecordDecl *RD) {
  if (!getTargetInfo().getCXXABI().hasKeyFunctions())
    return 0;

  assert(RD->getDefinition() && "Cannot get key function for forward decl!");
  RD = cast<CXXRecordDecl>(RD->getDefinition());

  LazyDeclPtr &Entry = KeyFunctions[RD];
  if (!Entry)
    Entry = const_cast<CXXMethodDecl*>(computeKeyFunction(*this, RD));

  return cast_or_null<CXXMethodDecl>(Entry.get(getExternalSource()));
}

void ASTContext::setNonKeyFunction(const CXXMethodDecl *Method) {
  assert(Method == Method->getFirstDecl() &&
         "not working with method declaration from class definition");

  // Look up the cache entry.  Since we're working with the first
  // declaration, its parent must be the class definition, which is
  // the correct key for the KeyFunctions hash.
  llvm::DenseMap<const CXXRecordDecl*, LazyDeclPtr>::iterator
    I = KeyFunctions.find(Method->getParent());

  // If it's not cached, there's nothing to do.
  if (I == KeyFunctions.end()) return;

  // If it is cached, check whether it's the target method, and if so,
  // remove it from the cache.
  if (I->second.get(getExternalSource()) == Method) {
    // FIXME: remember that we did this for module / chained PCH state?
    KeyFunctions.erase(I);
  }
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

static void PrintIndentNoOffset(raw_ostream &OS, unsigned IndentLevel) {
  OS << "     | ";
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
  bool HasOwnVFPtr = Layout.hasOwnVFPtr();
  bool HasOwnVBPtr = Layout.hasOwnVBPtr();

  // Vtable pointer.
  if (RD->isDynamicClass() && !PrimaryBase && !isMsLayout(RD)) {
    PrintOffset(OS, Offset, IndentLevel);
    OS << '(' << *RD << " vtable pointer)\n";
  } else if (HasOwnVFPtr) {
    PrintOffset(OS, Offset, IndentLevel);
    // vfptr (for Microsoft C++ ABI)
    OS << '(' << *RD << " vftable pointer)\n";
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

  // vbptr (for Microsoft C++ ABI)
  if (HasOwnVBPtr) {
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

  PrintIndentNoOffset(OS, IndentLevel - 1);
  OS << "[sizeof=" << Layout.getSize().getQuantity();
  if (!isMsLayout(RD))
    OS << ", dsize=" << Layout.getDataSize().getQuantity();
  OS << ", align=" << Layout.getAlignment().getQuantity() << '\n';

  PrintIndentNoOffset(OS, IndentLevel - 1);
  OS << " nvsize=" << Layout.getNonVirtualSize().getQuantity();
  OS << ", nvalign=" << Layout.getNonVirtualAlign().getQuantity() << "]\n";
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
  if (!isMsLayout(RD))
    OS << "  DataSize:" << toBits(Info.getDataSize()) << "\n";
  OS << "  Alignment:" << toBits(Info.getAlignment()) << "\n";
  OS << "  FieldOffsets: [";
  for (unsigned i = 0, e = Info.getFieldCount(); i != e; ++i) {
    if (i) OS << ", ";
    OS << Info.getFieldOffset(i);
  }
  OS << "]>\n";
}
