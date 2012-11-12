//===--- CGVTables.h - Emit LLVM Code for C++ vtables -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of virtual tables.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGVTABLE_H
#define CLANG_CODEGEN_CGVTABLE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/GlobalVariable.h"
#include "clang/Basic/ABI.h"
#include "clang/AST/BaseSubobject.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/VTableBuilder.h"

namespace clang {
  class CXXRecordDecl;

namespace CodeGen {
  class CodeGenModule;

class CodeGenVTables {
  CodeGenModule &CGM;

  VTableContext VTContext;

  /// VTables - All the vtables which have been defined.
  llvm::DenseMap<const CXXRecordDecl *, llvm::GlobalVariable *> VTables;
  
  /// VTableAddressPointsMapTy - Address points for a single vtable.
  typedef llvm::DenseMap<BaseSubobject, uint64_t> VTableAddressPointsMapTy;

  typedef std::pair<const CXXRecordDecl *, BaseSubobject> BaseSubobjectPairTy;
  typedef llvm::DenseMap<BaseSubobjectPairTy, uint64_t> SubVTTIndiciesMapTy;
  
  /// SubVTTIndicies - Contains indices into the various sub-VTTs.
  SubVTTIndiciesMapTy SubVTTIndicies;

  typedef llvm::DenseMap<BaseSubobjectPairTy, uint64_t>
    SecondaryVirtualPointerIndicesMapTy;

  /// SecondaryVirtualPointerIndices - Contains the secondary virtual pointer
  /// indices.
  SecondaryVirtualPointerIndicesMapTy SecondaryVirtualPointerIndices;

  /// EmitThunk - Emit a single thunk.
  void EmitThunk(GlobalDecl GD, const ThunkInfo &Thunk, 
                 bool UseAvailableExternallyLinkage);

  /// MaybeEmitThunkAvailableExternally - Try to emit the given thunk with
  /// available_externally linkage to allow for inlining of thunks.
  /// This will be done iff optimizations are enabled and the member function
  /// doesn't contain any incomplete types.
  void MaybeEmitThunkAvailableExternally(GlobalDecl GD, const ThunkInfo &Thunk);

  /// CreateVTableInitializer - Create a vtable initializer for the given record
  /// decl.
  /// \param Components - The vtable components; this is really an array of
  /// VTableComponents.
  llvm::Constant *CreateVTableInitializer(const CXXRecordDecl *RD,
                                          const VTableComponent *Components, 
                                          unsigned NumComponents,
                                const VTableLayout::VTableThunkTy *VTableThunks,
                                          unsigned NumVTableThunks);

public:
  CodeGenVTables(CodeGenModule &CGM);

  VTableContext &getVTableContext() { return VTContext; }

  /// \brief True if the VTable of this record must be emitted in the
  /// translation unit.
  bool ShouldEmitVTableInThisTU(const CXXRecordDecl *RD);

  /// needsVTTParameter - Return whether the given global decl needs a VTT
  /// parameter, which it does if it's a base constructor or destructor with
  /// virtual bases.
  static bool needsVTTParameter(GlobalDecl GD);

  /// getSubVTTIndex - Return the index of the sub-VTT for the base class of the
  /// given record decl.
  uint64_t getSubVTTIndex(const CXXRecordDecl *RD, BaseSubobject Base);
  
  /// getSecondaryVirtualPointerIndex - Return the index in the VTT where the
  /// virtual pointer for the given subobject is located.
  uint64_t getSecondaryVirtualPointerIndex(const CXXRecordDecl *RD,
                                           BaseSubobject Base);

  /// getAddressPoint - Get the address point of the given subobject in the
  /// class decl.
  uint64_t getAddressPoint(BaseSubobject Base, const CXXRecordDecl *RD);
  
  /// GetAddrOfVTable - Get the address of the vtable for the given record decl.
  llvm::GlobalVariable *GetAddrOfVTable(const CXXRecordDecl *RD);

  /// EmitVTableDefinition - Emit the definition of the given vtable.
  void EmitVTableDefinition(llvm::GlobalVariable *VTable,
                            llvm::GlobalVariable::LinkageTypes Linkage,
                            const CXXRecordDecl *RD);
  
  /// GenerateConstructionVTable - Generate a construction vtable for the given 
  /// base subobject.
  llvm::GlobalVariable *
  GenerateConstructionVTable(const CXXRecordDecl *RD, const BaseSubobject &Base, 
                             bool BaseIsVirtual, 
                             llvm::GlobalVariable::LinkageTypes Linkage,
                             VTableAddressPointsMapTy& AddressPoints);

    
  /// GetAddrOfVTable - Get the address of the VTT for the given record decl.
  llvm::GlobalVariable *GetAddrOfVTT(const CXXRecordDecl *RD);

  /// EmitVTTDefinition - Emit the definition of the given vtable.
  void EmitVTTDefinition(llvm::GlobalVariable *VTT,
                         llvm::GlobalVariable::LinkageTypes Linkage,
                         const CXXRecordDecl *RD);

  /// EmitThunks - Emit the associated thunks for the given global decl.
  void EmitThunks(GlobalDecl GD);
    
  /// GenerateClassData - Generate all the class data required to be generated
  /// upon definition of a KeyFunction.  This includes the vtable, the
  /// rtti data structure and the VTT.
  ///
  /// \param Linkage - The desired linkage of the vtable, the RTTI and the VTT.
  void GenerateClassData(llvm::GlobalVariable::LinkageTypes Linkage,
                         const CXXRecordDecl *RD);
};

} // end namespace CodeGen
} // end namespace clang
#endif
