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
