//===--- CIRGenTypes.h - Type translation for CIR CodeGen -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the code that handles AST -> CIR type lowering.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CODEGENTYPES_H
#define LLVM_CLANG_LIB_CODEGEN_CODEGENTYPES_H

#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "llvm/ADT/SmallPtrSet.h"

namespace clang {
class ASTContext;
class QualType;
class Type;
} // namespace clang

namespace mlir {
class Type;
}

namespace clang::CIRGen {

class CIRGenBuilderTy;
class CIRGenModule;

/// This class organizes the cross-module state that is used while lowering
/// AST types to CIR types.
class CIRGenTypes {
  CIRGenModule &cgm;
  clang::ASTContext &astContext;
  CIRGenBuilderTy &builder;

public:
  CIRGenTypes(CIRGenModule &cgm);
  ~CIRGenTypes();

  /// This map of clang::Type to mlir::Type (which includes CIR type) is a
  /// cache of types that have already been processed.
  using TypeCacheTy = llvm::DenseMap<const clang::Type *, mlir::Type>;
  TypeCacheTy typeCache;

  mlir::MLIRContext &getMLIRContext() const;

  /// Convert a Clang type into a mlir::Type.
  mlir::Type convertType(clang::QualType type);
};

} // namespace clang::CIRGen

#endif
