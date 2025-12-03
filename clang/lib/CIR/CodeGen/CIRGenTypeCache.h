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
#include "clang/Basic/AddressSpaces.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace clang::CIRGen {

/// This structure provides a set of types that are commonly used
/// during IR emission. It's initialized once in CodeGenModule's
/// constructor and then copied around into new CIRGenFunction's.
struct CIRGenTypeCache {
  CIRGenTypeCache() {}

  // ClangIR void type
  cir::VoidType voidTy;

  // ClangIR signed integral types of common sizes
  cir::IntType sInt8Ty;
  cir::IntType sInt16Ty;
  cir::IntType sInt32Ty;
  cir::IntType sInt64Ty;
  cir::IntType sInt128Ty;

  // ClangIR unsigned integral type of common sizes
  cir::IntType uInt8Ty;
  cir::IntType uInt16Ty;
  cir::IntType uInt32Ty;
  cir::IntType uInt64Ty;
  cir::IntType uInt128Ty;

  // ClangIR floating-point types with fixed formats
  cir::FP16Type fP16Ty;
  cir::BF16Type bFloat16Ty;
  cir::SingleType floatTy;
  cir::DoubleType doubleTy;
  cir::FP80Type fP80Ty;
  cir::FP128Type fP128Ty;

  /// ClangIR char
  mlir::Type uCharTy;

  /// intptr_t, size_t, and ptrdiff_t, which we assume are the same size.
  union {
    mlir::Type uIntPtrTy;
    mlir::Type sizeTy;
  };

  mlir::Type ptrDiffTy;

  /// void* in address space 0
  cir::PointerType voidPtrTy;
  cir::PointerType uInt8PtrTy;

  /// void* in alloca address space
  cir::PointerType allocaInt8PtrTy;

  /// The size and alignment of a pointer into the generic address space.
  union {
    unsigned char PointerAlignInBytes;
    unsigned char PointerSizeInBytes;
  };

  /// The size and alignment of size_t.
  union {
    unsigned char SizeSizeInBytes; // sizeof(size_t)
    unsigned char SizeAlignInBytes;
  };

  cir::TargetAddressSpaceAttr cirAllocaAddressSpace;

  clang::CharUnits getSizeSize() const {
    return clang::CharUnits::fromQuantity(SizeSizeInBytes);
  }
  clang::CharUnits getSizeAlign() const {
    return clang::CharUnits::fromQuantity(SizeAlignInBytes);
  }

  clang::CharUnits getPointerAlign() const {
    return clang::CharUnits::fromQuantity(PointerAlignInBytes);
  }

  cir::TargetAddressSpaceAttr getCIRAllocaAddressSpace() const {
    return cirAllocaAddressSpace;
  }
};

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENTYPECACHE_H
