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

#ifndef LLVM_CLANG_LIB_CIR_CODEGENTYPECACHE_H
#define LLVM_CLANG_LIB_CIR_CODEGENTYPECACHE_H

#include "UnimplementedFeatureGuarding.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "clang/AST/CharUnits.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace cir {

/// This structure provides a set of types that are commonly used
/// during IR emission. It's initialized once in CodeGenModule's
/// constructor and then copied around into new CIRGenFunction's.
struct CIRGenTypeCache {
  CIRGenTypeCache() {}

  /// void
  mlir::cir::VoidType VoidTy;
  // char, int, short, long
  mlir::cir::IntType SInt8Ty, SInt16Ty, SInt32Ty, SInt64Ty;
  // usigned char, unsigned, unsigned short, unsigned long
  mlir::cir::IntType UInt8Ty, UInt16Ty, UInt32Ty, UInt64Ty;
  /// half, bfloat, float, double
  // mlir::Type HalfTy, BFloatTy;
  // TODO(cir): perhaps we should abstract long double variations into a custom
  // cir.long_double type. Said type would also hold the semantics for lowering.
  mlir::cir::SingleType FloatTy;
  mlir::cir::DoubleType DoubleTy;
  mlir::cir::FP80Type FP80Ty;

  /// int
  mlir::Type UIntTy;

  /// char
  mlir::Type UCharTy;

  /// intptr_t, size_t, and ptrdiff_t, which we assume are the same size.
  union {
    mlir::Type UIntPtrTy;
    mlir::Type SizeTy;
  };

  mlir::Type PtrDiffTy;

  /// void* in address space 0
  mlir::cir::PointerType VoidPtrTy;
  mlir::cir::PointerType UInt8PtrTy;

  /// void** in address space 0
  union {
    mlir::cir::PointerType VoidPtrPtrTy;
    mlir::cir::PointerType UInt8PtrPtrTy;
  };

  /// void* in alloca address space
  union {
    mlir::cir::PointerType AllocaVoidPtrTy;
    mlir::cir::PointerType AllocaInt8PtrTy;
  };

  /// void* in default globals address space
  //   union {
  //     mlir::cir::PointerType GlobalsVoidPtrTy;
  //     mlir::cir::PointerType GlobalsInt8PtrTy;
  //   };

  /// void* in the address space for constant globals
  //   mlir::cir::PointerType ConstGlobalsPtrTy;

  /// The size and alignment of the builtin C type 'int'.  This comes
  /// up enough in various ABI lowering tasks to be worth pre-computing.
  //   union {
  //     unsigned char IntSizeInBytes;
  //     unsigned char IntAlignInBytes;
  //   };
  //   clang::CharUnits getIntSize() const {
  //     return clang::CharUnits::fromQuantity(IntSizeInBytes);
  //   }
  //   clang::CharUnits getIntAlign() const {
  //     return clang::CharUnits::fromQuantity(IntAlignInBytes);
  //   }

  /// The width of a pointer into the generic address space.
  //   unsigned char PointerWidthInBits;

  /// The size and alignment of a pointer into the generic address space.
  union {
    unsigned char PointerAlignInBytes;
    unsigned char PointerSizeInBytes;
  };

  /// The size and alignment of size_t.
  //   union {
  //     unsigned char SizeSizeInBytes; // sizeof(size_t)
  //     unsigned char SizeAlignInBytes;
  //   };

  clang::LangAS ASTAllocaAddressSpace;

  //   clang::CharUnits getSizeSize() const {
  //     return clang::CharUnits::fromQuantity(SizeSizeInBytes);
  //   }
  //   clang::CharUnits getSizeAlign() const {
  //     return clang::CharUnits::fromQuantity(SizeAlignInBytes);
  //   }
  clang::CharUnits getPointerSize() const {
    return clang::CharUnits::fromQuantity(PointerSizeInBytes);
  }
  clang::CharUnits getPointerAlign() const {
    return clang::CharUnits::fromQuantity(PointerAlignInBytes);
  }

  clang::LangAS getASTAllocaAddressSpace() const {
    // Address spaces are not yet fully supported, but the usage of the default
    // alloca address space can be used for now only for comparison with the
    // default address space.
    assert(!UnimplementedFeature::addressSpace());
    assert(ASTAllocaAddressSpace == clang::LangAS::Default);
    return ASTAllocaAddressSpace;
  }
};

} // namespace cir

#endif
