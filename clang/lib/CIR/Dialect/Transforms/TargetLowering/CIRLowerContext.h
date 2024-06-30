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

namespace mlir {
namespace cir {

// FIXME(cir): Most of this is type-related information that should already be
// embedded into CIR. Maybe we can move this to an MLIR interface.
class CIRLowerContext : public llvm::RefCountedBase<CIRLowerContext> {

private:
  mutable SmallVector<Type, 0> Types;

  clang::TypeInfo getTypeInfoImpl(const Type T) const;

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
  CIRLowerContext(ModuleOp module, clang::LangOptions &LOpts);
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

  //===--------------------------------------------------------------------===//
  //                         Type Sizing and Analysis
  //===--------------------------------------------------------------------===//

  /// Get the size and alignment of the specified complete type in bits.
  clang::TypeInfo getTypeInfo(Type T) const;

  /// Return the size of the specified (complete) type \p T, in bits.
  uint64_t getTypeSize(Type T) const { return getTypeInfo(T).Width; }

  /// Return the size of the character type, in bits.
  // FIXME(cir): Refactor types and properly implement DataLayout interface in
  // CIR so that this can be queried from the module.
  uint64_t getCharWidth() const { return 8; }

  /// Convert a size in bits to a size in characters.
  clang::CharUnits toCharUnitsFromBits(int64_t BitSize) const;

  clang::CharUnits getTypeSizeInChars(Type T) const {
    // FIXME(cir): We should query MLIR's Datalayout here instead.
    return getTypeInfoInChars(T).Width;
  }

  /// Return the ABI-specified alignment of a (complete) type \p T, in
  /// bits.
  unsigned getTypeAlign(Type T) const { return getTypeInfo(T).Align; }

  clang::TypeInfoChars getTypeInfoInChars(Type T) const;

  /// More type predicates useful for type checking/promotion
  bool isPromotableIntegerType(Type T) const; // C99 6.3.1.1p2
};

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRLowerContext_H
