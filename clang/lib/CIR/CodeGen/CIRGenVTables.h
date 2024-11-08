//===--- CIRGenVTables.h - Emit LLVM Code for C++ vtables -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of virtual tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENVTABLES_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENVTABLES_H

#include "ConstantInitBuilder.h"
#include "clang/AST/BaseSubobject.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/VTableBuilder.h"
#include "clang/Basic/ABI.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
class CXXRecordDecl;
}

namespace clang::CIRGen {
class CIRGenModule;

class CIRGenVTables {
  CIRGenModule &CGM;

  clang::VTableContextBase *VTContext;

  /// Address points for a single vtable.
  typedef clang::VTableLayout::AddressPointsMapTy VTableAddressPointsMapTy;

  typedef std::pair<const clang::CXXRecordDecl *, clang::BaseSubobject>
      BaseSubobjectPairTy;
  typedef llvm::DenseMap<BaseSubobjectPairTy, uint64_t> SubVTTIndiciesMapTy;

  /// Contains indices into the various sub-VTTs.
  SubVTTIndiciesMapTy SubVTTIndicies;

  typedef llvm::DenseMap<BaseSubobjectPairTy, uint64_t>
      SecondaryVirtualPointerIndicesMapTy;

  /// Contains the secondary virtual pointer
  /// indices.
  SecondaryVirtualPointerIndicesMapTy SecondaryVirtualPointerIndices;

  /// Cache for the pure virtual member call function.
  mlir::cir::FuncOp PureVirtualFn = nullptr;

  /// Cache for the deleted virtual member call function.
  mlir::cir::FuncOp DeletedVirtualFn = nullptr;

  void addVTableComponent(ConstantArrayBuilder &builder,
                          const VTableLayout &layout, unsigned componentIndex,
                          mlir::Attribute rtti, unsigned &nextVTableThunkIndex,
                          unsigned vtableAddressPoint,
                          bool vtableHasLocalLinkage);

  bool useRelativeLayout() const;
  mlir::Type getVTableComponentType();

public:
  /// Add vtable components for the given vtable layout to the given
  /// global initializer.
  void createVTableInitializer(ConstantStructBuilder &builder,
                               const VTableLayout &layout, mlir::Attribute rtti,
                               bool vtableHasLocalLinkage);

  CIRGenVTables(CIRGenModule &CGM);

  clang::ItaniumVTableContext &getItaniumVTableContext() {
    return *llvm::cast<clang::ItaniumVTableContext>(VTContext);
  }

  const clang::ItaniumVTableContext &getItaniumVTableContext() const {
    return *llvm::cast<clang::ItaniumVTableContext>(VTContext);
  }

  /// Return the index of the sub-VTT for the base class of the given record
  /// decl.
  uint64_t getSubVTTIndex(const CXXRecordDecl *RD, BaseSubobject Base);

  /// Return the index in the VTT where the virtual pointer for the given
  /// subobject is located.
  uint64_t getSecondaryVirtualPointerIndex(const CXXRecordDecl *RD,
                                           BaseSubobject Base);

  /// Generate a construction vtable for the given base subobject.
  mlir::cir::GlobalOp
  generateConstructionVTable(const CXXRecordDecl *RD, const BaseSubobject &Base,
                             bool BaseIsVirtual,
                             mlir::cir::GlobalLinkageKind Linkage,
                             VTableAddressPointsMapTy &AddressPoints);

  /// Get the address of the VTT for the given record decl.
  mlir::cir::GlobalOp getAddrOfVTT(const CXXRecordDecl *RD);

  /// Emit the definition of the given vtable.
  void buildVTTDefinition(mlir::cir::GlobalOp VTT,
                          mlir::cir::GlobalLinkageKind Linkage,
                          const CXXRecordDecl *RD);

  /// Emit the associated thunks for the given global decl.
  void buildThunks(GlobalDecl GD);

  /// Generate all the class data required to be generated upon definition of a
  /// KeyFunction. This includes the vtable, the RTTI data structure (if RTTI
  /// is enabled) and the VTT (if the class has virtual bases).
  void GenerateClassData(const clang::CXXRecordDecl *RD);

  bool isVTableExternal(const clang::CXXRecordDecl *RD);

  /// Returns the type of a vtable with the given layout. Normally a struct of
  /// arrays of pointers, with one struct element for each vtable in the vtable
  /// group.
  mlir::Type getVTableType(const clang::VTableLayout &layout);
};

} // namespace clang::CIRGen
#endif
