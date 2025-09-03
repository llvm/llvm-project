//===----------------------------------------------------------------------===//
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

#ifndef CLANG_LIB_CIR_CODEGEN_CIRGENVTABLES_H
#define CLANG_LIB_CIR_CODEGEN_CIRGENVTABLES_H

#include "mlir/IR/Types.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/VTableBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace clang {
class CXXRecordDecl;
}

namespace clang::CIRGen {
class CIRGenModule;

class CIRGenVTables {
  CIRGenModule &cgm;

  clang::VTableContextBase *vtContext;

  /// Address points for a single vtable.
  using VTableAddressPointsMapTy = clang::VTableLayout::AddressPointsMapTy;
  using BaseSubobjectPairTy =
      std::pair<const clang::CXXRecordDecl *, clang::BaseSubobject>;
  using SubVTTIndiciesMapTy = llvm::DenseMap<BaseSubobjectPairTy, uint64_t>;

  /// Contains indices into the various sub-VTTs.
  SubVTTIndiciesMapTy subVTTIndicies;

  using SecondaryVirtualPointerIndicesMapTy =
      llvm::DenseMap<BaseSubobjectPairTy, uint64_t>;

  /// Contains the secondary virtual pointer
  /// indices.
  SecondaryVirtualPointerIndicesMapTy secondaryVirtualPointerIndices;

  mlir::Attribute
  getVTableComponent(const VTableLayout &layout, unsigned componentIndex,
                     mlir::Attribute rtti, unsigned &nextVTableThunkIndex,
                     unsigned vtableAddressPoint, bool vtableHasLocalLinkage);

  mlir::Type getVTableComponentType();

public:
  CIRGenVTables(CIRGenModule &cgm);

  /// Add vtable components for the given vtable layout to the given
  /// global initializer.
  void createVTableInitializer(cir::GlobalOp &vtable,
                               const clang::VTableLayout &layout,
                               mlir::Attribute rtti,
                               bool vtableHasLocalLinkage);

  clang::ItaniumVTableContext &getItaniumVTableContext() {
    return *llvm::cast<clang::ItaniumVTableContext>(vtContext);
  }

  const clang::ItaniumVTableContext &getItaniumVTableContext() const {
    return *llvm::cast<clang::ItaniumVTableContext>(vtContext);
  }

  /// Generate a construction vtable for the given base subobject.
  cir::GlobalOp
  generateConstructionVTable(const CXXRecordDecl *rd, const BaseSubobject &base,
                             bool baseIsVirtual, cir::GlobalLinkageKind linkage,
                             VTableAddressPointsMapTy &addressPoints);

  /// Get the address of the VTT for the given record decl.
  cir::GlobalOp getAddrOfVTT(const CXXRecordDecl *rd);

  /// Emit the definition of the given vtable.
  void emitVTTDefinition(cir::GlobalOp vttOp, cir::GlobalLinkageKind linkage,
                         const CXXRecordDecl *rd);
  /// Return the index of the sub-VTT for the base class of the given record
  /// decl.
  uint64_t getSubVTTIndex(const CXXRecordDecl *rd, BaseSubobject base);

  /// Return the index in the VTT where the virtual pointer for the given
  /// subobject is located.
  uint64_t getSecondaryVirtualPointerIndex(const CXXRecordDecl *rd,
                                           BaseSubobject base);

  /// Emit the associated thunks for the given global decl.
  void emitThunks(GlobalDecl gd);

  /// Generate all the class data required to be generated upon definition of a
  /// KeyFunction. This includes the vtable, the RTTI data structure (if RTTI
  /// is enabled) and the VTT (if the class has virtual bases).
  void generateClassData(const CXXRecordDecl *rd);

  /// Returns the type of a vtable with the given layout. Normally a struct of
  /// arrays of pointers, with one struct element for each vtable in the vtable
  /// group.
  cir::RecordType getVTableType(const clang::VTableLayout &layout);
};

} // namespace clang::CIRGen

#endif // CLANG_LIB_CIR_CODEGEN_CIRGENVTABLES_H
