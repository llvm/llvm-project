//===--- VTTBuilder.h - C++ VTT layout builder --------------------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with generation of the layout of virtual table
// tables (VTT).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_VTTBUILDER_H
#define LLVM_CLANG_AST_VTTBUILDER_H

#include "clang/AST/BaseSubobject.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/ABI.h"
#include <utility>

namespace clang {

class VTTVTable {
  llvm::PointerIntPair<const CXXRecordDecl *, 1, bool> BaseAndIsVirtual;
  CharUnits BaseOffset;

public:
  VTTVTable() {}
  VTTVTable(const CXXRecordDecl *Base, CharUnits BaseOffset, bool BaseIsVirtual)
    : BaseAndIsVirtual(Base, BaseIsVirtual), BaseOffset(BaseOffset) {}
  VTTVTable(BaseSubobject Base, bool BaseIsVirtual)
    : BaseAndIsVirtual(Base.getBase(), BaseIsVirtual),
      BaseOffset(Base.getBaseOffset()) {}

  const CXXRecordDecl *getBase() const {
    return BaseAndIsVirtual.getPointer();
  }

  CharUnits getBaseOffset() const {
    return BaseOffset;
  }

  bool isVirtual() const {
    return BaseAndIsVirtual.getInt();
  }

  BaseSubobject getBaseSubobject() const {
    return BaseSubobject(getBase(), getBaseOffset());
  }
};

struct VTTComponent {
  uint64_t VTableIndex;
  BaseSubobject VTableBase;

  VTTComponent() {}
  VTTComponent(uint64_t VTableIndex, BaseSubobject VTableBase)
    : VTableIndex(VTableIndex), VTableBase(VTableBase) {}
};

/// \brief Class for building VTT layout information.
class VTTBuilder {
  
  ASTContext &Ctx;

  /// \brief The most derived class for which we're building this vtable.
  const CXXRecordDecl *MostDerivedClass;

  typedef SmallVector<VTTVTable, 64> VTTVTablesVectorTy;
  
  /// \brief The VTT vtables.
  VTTVTablesVectorTy VTTVTables;
  
  typedef SmallVector<VTTComponent, 64> VTTComponentsVectorTy;
  
  /// \brief The VTT components.
  VTTComponentsVectorTy VTTComponents;
  
  /// \brief The AST record layout of the most derived class.
  const ASTRecordLayout &MostDerivedClassLayout;

  typedef llvm::SmallPtrSet<const CXXRecordDecl *, 4> VisitedVirtualBasesSetTy;

  typedef llvm::DenseMap<BaseSubobject, uint64_t> AddressPointsMapTy;

  /// \brief The sub-VTT indices for the bases of the most derived class.
  llvm::DenseMap<BaseSubobject, uint64_t> SubVTTIndicies;

  /// \brief The secondary virtual pointer indices of all subobjects of
  /// the most derived class.
  llvm::DenseMap<BaseSubobject, uint64_t> SecondaryVirtualPointerIndices;

  /// \brief Whether the VTT builder should generate LLVM IR for the VTT.
  bool GenerateDefinition;

  /// \brief Add a vtable pointer to the VTT currently being built.
  void AddVTablePointer(BaseSubobject Base, uint64_t VTableIndex,
                        const CXXRecordDecl *VTableClass);
                        
  /// \brief Lay out the secondary VTTs of the given base subobject.
  void LayoutSecondaryVTTs(BaseSubobject Base);
  
  /// \brief Lay out the secondary virtual pointers for the given base
  /// subobject.
  ///
  /// \param BaseIsMorallyVirtual whether the base subobject is a virtual base
  /// or a direct or indirect base of a virtual base.
  void LayoutSecondaryVirtualPointers(BaseSubobject Base, 
                                      bool BaseIsMorallyVirtual,
                                      uint64_t VTableIndex,
                                      const CXXRecordDecl *VTableClass,
                                      VisitedVirtualBasesSetTy &VBases);
  
  /// \brief Lay out the secondary virtual pointers for the given base
  /// subobject.
  void LayoutSecondaryVirtualPointers(BaseSubobject Base, 
                                      uint64_t VTableIndex);

  /// \brief Lay out the VTTs for the virtual base classes of the given
  /// record declaration.
  void LayoutVirtualVTTs(const CXXRecordDecl *RD,
                         VisitedVirtualBasesSetTy &VBases);
  
  /// \brief Lay out the VTT for the given subobject, including any
  /// secondary VTTs, secondary virtual pointers and virtual VTTs.
  void LayoutVTT(BaseSubobject Base, bool BaseIsVirtual);
  
public:
  VTTBuilder(ASTContext &Ctx, const CXXRecordDecl *MostDerivedClass,
             bool GenerateDefinition);

  // \brief Returns a reference to the VTT components.
  const VTTComponentsVectorTy &getVTTComponents() const {
    return VTTComponents;
  }
  
  // \brief Returns a reference to the VTT vtables.
  const VTTVTablesVectorTy &getVTTVTables() const {
    return VTTVTables;
  }
  
  /// \brief Returns a reference to the sub-VTT indices.
  const llvm::DenseMap<BaseSubobject, uint64_t> &getSubVTTIndicies() const {
    return SubVTTIndicies;
  }
  
  /// \brief Returns a reference to the secondary virtual pointer indices.
  const llvm::DenseMap<BaseSubobject, uint64_t> &
  getSecondaryVirtualPointerIndices() const {
    return SecondaryVirtualPointerIndices;
  }

};

}

#endif
