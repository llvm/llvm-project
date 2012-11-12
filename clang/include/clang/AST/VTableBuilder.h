//===--- VTableBuilder.h - C++ vtable layout builder --------------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with generation of the layout of virtual tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_VTABLEBUILDER_H
#define LLVM_CLANG_AST_VTABLEBUILDER_H

#include "clang/AST/BaseSubobject.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/ABI.h"
#include "llvm/ADT/SetVector.h"
#include <utility>

namespace clang {
  class CXXRecordDecl;

/// VTableComponent - Represents a single component in a vtable.
class VTableComponent {
public:
  enum Kind {
    CK_VCallOffset,
    CK_VBaseOffset,
    CK_OffsetToTop,
    CK_RTTI,
    CK_FunctionPointer,

    /// CK_CompleteDtorPointer - A pointer to the complete destructor.
    CK_CompleteDtorPointer,

    /// CK_DeletingDtorPointer - A pointer to the deleting destructor.
    CK_DeletingDtorPointer,

    /// CK_UnusedFunctionPointer - In some cases, a vtable function pointer
    /// will end up never being called. Such vtable function pointers are
    /// represented as a CK_UnusedFunctionPointer.
    CK_UnusedFunctionPointer
  };

  VTableComponent() { }

  static VTableComponent MakeVCallOffset(CharUnits Offset) {
    return VTableComponent(CK_VCallOffset, Offset);
  }

  static VTableComponent MakeVBaseOffset(CharUnits Offset) {
    return VTableComponent(CK_VBaseOffset, Offset);
  }

  static VTableComponent MakeOffsetToTop(CharUnits Offset) {
    return VTableComponent(CK_OffsetToTop, Offset);
  }

  static VTableComponent MakeRTTI(const CXXRecordDecl *RD) {
    return VTableComponent(CK_RTTI, reinterpret_cast<uintptr_t>(RD));
  }

  static VTableComponent MakeFunction(const CXXMethodDecl *MD) {
    assert(!isa<CXXDestructorDecl>(MD) &&
           "Don't use MakeFunction with destructors!");

    return VTableComponent(CK_FunctionPointer,
                           reinterpret_cast<uintptr_t>(MD));
  }

  static VTableComponent MakeCompleteDtor(const CXXDestructorDecl *DD) {
    return VTableComponent(CK_CompleteDtorPointer,
                           reinterpret_cast<uintptr_t>(DD));
  }

  static VTableComponent MakeDeletingDtor(const CXXDestructorDecl *DD) {
    return VTableComponent(CK_DeletingDtorPointer,
                           reinterpret_cast<uintptr_t>(DD));
  }

  static VTableComponent MakeUnusedFunction(const CXXMethodDecl *MD) {
    assert(!isa<CXXDestructorDecl>(MD) &&
           "Don't use MakeUnusedFunction with destructors!");
    return VTableComponent(CK_UnusedFunctionPointer,
                           reinterpret_cast<uintptr_t>(MD));
  }

  static VTableComponent getFromOpaqueInteger(uint64_t I) {
    return VTableComponent(I);
  }

  /// getKind - Get the kind of this vtable component.
  Kind getKind() const {
    return (Kind)(Value & 0x7);
  }

  CharUnits getVCallOffset() const {
    assert(getKind() == CK_VCallOffset && "Invalid component kind!");

    return getOffset();
  }

  CharUnits getVBaseOffset() const {
    assert(getKind() == CK_VBaseOffset && "Invalid component kind!");

    return getOffset();
  }

  CharUnits getOffsetToTop() const {
    assert(getKind() == CK_OffsetToTop && "Invalid component kind!");

    return getOffset();
  }

  const CXXRecordDecl *getRTTIDecl() const {
    assert(getKind() == CK_RTTI && "Invalid component kind!");

    return reinterpret_cast<CXXRecordDecl *>(getPointer());
  }

  const CXXMethodDecl *getFunctionDecl() const {
    assert(getKind() == CK_FunctionPointer);

    return reinterpret_cast<CXXMethodDecl *>(getPointer());
  }

  const CXXDestructorDecl *getDestructorDecl() const {
    assert((getKind() == CK_CompleteDtorPointer ||
            getKind() == CK_DeletingDtorPointer) && "Invalid component kind!");

    return reinterpret_cast<CXXDestructorDecl *>(getPointer());
  }

  const CXXMethodDecl *getUnusedFunctionDecl() const {
    assert(getKind() == CK_UnusedFunctionPointer);

    return reinterpret_cast<CXXMethodDecl *>(getPointer());
  }

private:
  VTableComponent(Kind ComponentKind, CharUnits Offset) {
    assert((ComponentKind == CK_VCallOffset ||
            ComponentKind == CK_VBaseOffset ||
            ComponentKind == CK_OffsetToTop) && "Invalid component kind!");
    assert(Offset.getQuantity() < (1LL << 56) && "Offset is too big!");
    assert(Offset.getQuantity() >= -(1LL << 56) && "Offset is too small!");

    Value = (uint64_t(Offset.getQuantity()) << 3) | ComponentKind;
  }

  VTableComponent(Kind ComponentKind, uintptr_t Ptr) {
    assert((ComponentKind == CK_RTTI ||
            ComponentKind == CK_FunctionPointer ||
            ComponentKind == CK_CompleteDtorPointer ||
            ComponentKind == CK_DeletingDtorPointer ||
            ComponentKind == CK_UnusedFunctionPointer) &&
            "Invalid component kind!");

    assert((Ptr & 7) == 0 && "Pointer not sufficiently aligned!");

    Value = Ptr | ComponentKind;
  }

  CharUnits getOffset() const {
    assert((getKind() == CK_VCallOffset || getKind() == CK_VBaseOffset ||
            getKind() == CK_OffsetToTop) && "Invalid component kind!");

    return CharUnits::fromQuantity(Value >> 3);
  }

  uintptr_t getPointer() const {
    assert((getKind() == CK_RTTI ||
            getKind() == CK_FunctionPointer ||
            getKind() == CK_CompleteDtorPointer ||
            getKind() == CK_DeletingDtorPointer ||
            getKind() == CK_UnusedFunctionPointer) &&
           "Invalid component kind!");

    return static_cast<uintptr_t>(Value & ~7ULL);
  }

  explicit VTableComponent(uint64_t Value)
    : Value(Value) { }

  /// The kind is stored in the lower 3 bits of the value. For offsets, we
  /// make use of the facts that classes can't be larger than 2^55 bytes,
  /// so we store the offset in the lower part of the 61 bytes that remain.
  /// (The reason that we're not simply using a PointerIntPair here is that we
  /// need the offsets to be 64-bit, even when on a 32-bit machine).
  int64_t Value;
};

class VTableLayout {
public:
  typedef std::pair<uint64_t, ThunkInfo> VTableThunkTy;
  typedef SmallVector<ThunkInfo, 1> ThunkInfoVectorTy;

  typedef const VTableComponent *vtable_component_iterator;
  typedef const VTableThunkTy *vtable_thunk_iterator;

  typedef llvm::DenseMap<BaseSubobject, uint64_t> AddressPointsMapTy;
private:
  uint64_t NumVTableComponents;
  llvm::OwningArrayPtr<VTableComponent> VTableComponents;

  /// VTableThunks - Contains thunks needed by vtables.
  uint64_t NumVTableThunks;
  llvm::OwningArrayPtr<VTableThunkTy> VTableThunks;

  /// Address points - Address points for all vtables.
  AddressPointsMapTy AddressPoints;

public:
  VTableLayout(uint64_t NumVTableComponents,
               const VTableComponent *VTableComponents,
               uint64_t NumVTableThunks,
               const VTableThunkTy *VTableThunks,
               const AddressPointsMapTy &AddressPoints);
  ~VTableLayout();

  uint64_t getNumVTableComponents() const {
    return NumVTableComponents;
  }

  vtable_component_iterator vtable_component_begin() const {
   return VTableComponents.get();
  }

  vtable_component_iterator vtable_component_end() const {
   return VTableComponents.get()+NumVTableComponents;
  }

  uint64_t getNumVTableThunks() const {
    return NumVTableThunks;
  }

  vtable_thunk_iterator vtable_thunk_begin() const {
   return VTableThunks.get();
  }

  vtable_thunk_iterator vtable_thunk_end() const {
   return VTableThunks.get()+NumVTableThunks;
  }

  uint64_t getAddressPoint(BaseSubobject Base) const {
    assert(AddressPoints.count(Base) &&
           "Did not find address point!");

    uint64_t AddressPoint = AddressPoints.lookup(Base);
    assert(AddressPoint && "Address point must not be zero!");

    return AddressPoint;
  }

  const AddressPointsMapTy &getAddressPoints() const {
    return AddressPoints;
  }
};

class VTableContext {
  ASTContext &Context;

public:
  typedef SmallVector<std::pair<uint64_t, ThunkInfo>, 1>
    VTableThunksTy;
  typedef SmallVector<ThunkInfo, 1> ThunkInfoVectorTy;

private:
  /// MethodVTableIndices - Contains the index (relative to the vtable address
  /// point) where the function pointer for a virtual function is stored.
  typedef llvm::DenseMap<GlobalDecl, int64_t> MethodVTableIndicesTy;
  MethodVTableIndicesTy MethodVTableIndices;

  typedef llvm::DenseMap<const CXXRecordDecl *, const VTableLayout *>
    VTableLayoutMapTy;
  VTableLayoutMapTy VTableLayouts;

  /// NumVirtualFunctionPointers - Contains the number of virtual function
  /// pointers in the vtable for a given record decl.
  llvm::DenseMap<const CXXRecordDecl *, uint64_t> NumVirtualFunctionPointers;

  typedef std::pair<const CXXRecordDecl *,
                    const CXXRecordDecl *> ClassPairTy;

  /// VirtualBaseClassOffsetOffsets - Contains the vtable offset (relative to
  /// the address point) in chars where the offsets for virtual bases of a class
  /// are stored.
  typedef llvm::DenseMap<ClassPairTy, CharUnits>
    VirtualBaseClassOffsetOffsetsMapTy;
  VirtualBaseClassOffsetOffsetsMapTy VirtualBaseClassOffsetOffsets;

  typedef llvm::DenseMap<const CXXMethodDecl *, ThunkInfoVectorTy> ThunksMapTy;

  /// Thunks - Contains all thunks that a given method decl will need.
  ThunksMapTy Thunks;

  void ComputeMethodVTableIndices(const CXXRecordDecl *RD);

  /// ComputeVTableRelatedInformation - Compute and store all vtable related
  /// information (vtable layout, vbase offset offsets, thunks etc) for the
  /// given record decl.
  void ComputeVTableRelatedInformation(const CXXRecordDecl *RD);

public:
  VTableContext(ASTContext &Context) : Context(Context) {}
  ~VTableContext();

  const VTableLayout &getVTableLayout(const CXXRecordDecl *RD) {
    ComputeVTableRelatedInformation(RD);
    assert(VTableLayouts.count(RD) && "No layout for this record decl!");

    return *VTableLayouts[RD];
  }

  VTableLayout *
  createConstructionVTableLayout(const CXXRecordDecl *MostDerivedClass,
                                 CharUnits MostDerivedClassOffset,
                                 bool MostDerivedClassIsVirtual,
                                 const CXXRecordDecl *LayoutClass);

  const ThunkInfoVectorTy *getThunkInfo(const CXXMethodDecl *MD) {
    ComputeVTableRelatedInformation(MD->getParent());

    ThunksMapTy::const_iterator I = Thunks.find(MD);
    if (I == Thunks.end()) {
      // We did not find a thunk for this method.
      return 0;
    }

    return &I->second;
  }

  /// getNumVirtualFunctionPointers - Return the number of virtual function
  /// pointers in the vtable for a given record decl.
  uint64_t getNumVirtualFunctionPointers(const CXXRecordDecl *RD);

  /// getMethodVTableIndex - Return the index (relative to the vtable address
  /// point) where the function pointer for the given virtual function is
  /// stored.
  uint64_t getMethodVTableIndex(GlobalDecl GD);

  /// getVirtualBaseOffsetOffset - Return the offset in chars (relative to the
  /// vtable address point) where the offset of the virtual base that contains
  /// the given base is stored, otherwise, if no virtual base contains the given
  /// class, return 0.  Base must be a virtual base class or an unambigious
  /// base.
  CharUnits getVirtualBaseOffsetOffset(const CXXRecordDecl *RD,
                                       const CXXRecordDecl *VBase);
};

}

#endif
