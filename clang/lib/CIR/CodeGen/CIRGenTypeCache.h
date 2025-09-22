//===--- CIRGenTypeCache.h - Commonly used LLVM types and info -*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This structure provides a set of common types useful during CIR emission.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENTYPECACHE_H
#define LLVM_CLANG_LIB_CIR_CIRGENTYPECACHE_H

#include "clang/AST/CharUnits.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace clang::CIRGen {

/// This structure provides a set of types that are commonly used
/// during IR emission. It's initialized once in CodeGenModule's
/// constructor and then copied around into new CIRGenFunction's.
struct CIRGenTypeCache {
  CIRGenTypeCache() {}

  // ClangIR void type
  cir::VoidType VoidTy;

  // ClangIR signed integral types of common sizes
  cir::IntType SInt8Ty;
  cir::IntType SInt16Ty;
  cir::IntType SInt32Ty;
  cir::IntType SInt64Ty;
  cir::IntType SInt128Ty;

  // ClangIR unsigned integral type of common sizes
  cir::IntType UInt8Ty;
  cir::IntType UInt16Ty;
  cir::IntType UInt32Ty;
  cir::IntType UInt64Ty;
  cir::IntType UInt128Ty;

  // ClangIR floating-point types with fixed formats
  cir::FP16Type FP16Ty;
  cir::BF16Type BFloat16Ty;
  cir::SingleType FloatTy;
  cir::DoubleType DoubleTy;
  cir::FP80Type FP80Ty;
  cir::FP128Type FP128Ty;

  /// ClangIR char
  mlir::Type UCharTy;

  /// intptr_t, size_t, and ptrdiff_t, which we assume are the same size.
  union {
    mlir::Type UIntPtrTy;
    mlir::Type SizeTy;
  };

  mlir::Type PtrDiffTy;

  /// void* in address space 0
  cir::PointerType VoidPtrTy;
  cir::PointerType UInt8PtrTy;

  /// The size and alignment of a pointer into the generic address space.
  union {
    unsigned char PointerAlignInBytes;
    unsigned char PointerSizeInBytes;
  };

  /// The alignment of size_t.
  unsigned char SizeAlignInBytes;

  clang::CharUnits getSizeAlign() const {
    return clang::CharUnits::fromQuantity(SizeAlignInBytes);
  }

  clang::CharUnits getPointerAlign() const {
    return clang::CharUnits::fromQuantity(PointerAlignInBytes);
  }
};

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENTYPECACHE_H
