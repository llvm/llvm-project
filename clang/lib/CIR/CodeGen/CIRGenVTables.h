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

namespace clang {
class CXXRecordDecl;
}

namespace clang::CIRGen {
class CIRGenModule;

class CIRGenVTables {
  CIRGenModule &cgm;

  clang::VTableContextBase *vtContext;

  mlir::Type getVTableComponentType();

public:
  CIRGenVTables(CIRGenModule &cgm);

  clang::ItaniumVTableContext &getItaniumVTableContext() {
    return *llvm::cast<clang::ItaniumVTableContext>(vtContext);
  }

  const clang::ItaniumVTableContext &getItaniumVTableContext() const {
    return *llvm::cast<clang::ItaniumVTableContext>(vtContext);
  }

  /// Returns the type of a vtable with the given layout. Normally a struct of
  /// arrays of pointers, with one struct element for each vtable in the vtable
  /// group.
  mlir::Type getVTableType(const clang::VTableLayout &layout);
};

} // namespace clang::CIRGen

#endif // CLANG_LIB_CIR_CODEGEN_CIRGENVTABLES_H
