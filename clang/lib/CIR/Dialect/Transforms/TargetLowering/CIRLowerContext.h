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
#include "clang/AST/Type.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace mlir {
namespace cir {

// FIXME(cir): Most of this is type-related information that should already be
// embedded into CIR. Maybe we can move this to an MLIR interface.
class CIRLowerContext : public llvm::RefCountedBase<CIRLowerContext> {

private:
  mutable SmallVector<Type, 0> Types;

  const clang::TargetInfo *Target = nullptr;
  const clang::TargetInfo *AuxTarget = nullptr;

  /// MLIR context to be used when creating types.
  MLIRContext *MLIRCtx;

  /// The language options used to create the AST associated with
  /// this ASTContext object.
  clang::LangOptions &LangOpts;

  //===--------------------------------------------------------------------===//
  //                         Built-in Types
  //===--------------------------------------------------------------------===//

  Type CharTy;

public:
  CIRLowerContext(MLIRContext *MLIRCtx, clang::LangOptions &LOpts);
  CIRLowerContext(const CIRLowerContext &) = delete;
  CIRLowerContext &operator=(const CIRLowerContext &) = delete;
  ~CIRLowerContext();

  /// Initialize built-in types.
  ///
  /// This routine may only be invoked once for a given ASTContext object.
  /// It is normally invoked after ASTContext construction.
  ///
  /// \param Target The target
  void initBuiltinTypes(const clang::TargetInfo &Target,
                        const clang::TargetInfo *AuxTarget = nullptr);

private:
  Type initBuiltinType(clang::BuiltinType::Kind K);

public:
  MLIRContext *getMLIRContext() const { return MLIRCtx; }
};

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRLowerContext_H
