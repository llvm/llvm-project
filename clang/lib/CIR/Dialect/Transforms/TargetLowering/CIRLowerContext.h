//===- CIRLowerContext.h - Context to lower CIR -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Partially mimics AST/ASTContext.h. The main difference is that this is
// adapted to operate on the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRLowerContext_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRLowerContext_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace cir {

// FIXME(cir): Most of this is type-related information that should already be
// embedded into CIR. Maybe we can move this to an MLIR interface.
class CIRLowerContext : public llvm::RefCountedBase<CIRLowerContext> {

private:
  mutable llvm::SmallVector<mlir::Type, 0> types;

  const clang::TargetInfo *target = nullptr;
  const clang::TargetInfo *auxTarget = nullptr;

  /// MLIR context to be used when creating types.
  mlir::MLIRContext *mlirContext;

  /// The language options used to create the AST associated with
  /// this ASTContext object.
  clang::LangOptions langOpts;

  /// Options for code generation.
  clang::CodeGenOptions codeGenOpts;

  //===--------------------------------------------------------------------===//
  //                         Built-in Types
  //===--------------------------------------------------------------------===//

  mlir::Type charTy;

public:
  CIRLowerContext(mlir::ModuleOp module, clang::LangOptions langOpts,
                  clang::CodeGenOptions codeGenOpts);
  ~CIRLowerContext();

  /// Initialize built-in types.
  ///
  /// This routine may only be invoked once for a given ASTContext object.
  /// It is normally invoked after ASTContext construction.
  ///
  /// \param Target The target
  void initBuiltinTypes(const clang::TargetInfo &target,
                        const clang::TargetInfo *auxTarget = nullptr);

private:
  mlir::Type initBuiltinType(clang::BuiltinType::Kind builtinKind);

public:
  mlir::MLIRContext *getMLIRContext() const { return mlirContext; }
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRLowerContext_H