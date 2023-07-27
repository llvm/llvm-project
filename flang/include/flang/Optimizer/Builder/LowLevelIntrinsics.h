//===-- LowLevelIntrinsics.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_OPTIMIZER_BUILDER_LOWLEVELINTRINSICS_H
#define FLANG_OPTIMIZER_BUILDER_LOWLEVELINTRINSICS_H

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir
namespace fir {
class FirOpBuilder;
}

namespace fir::factory {

/// Get the LLVM intrinsic for `memcpy`. Use the 64 bit version.
mlir::func::FuncOp getLlvmMemcpy(FirOpBuilder &builder);

/// Get the LLVM intrinsic for `memmove`. Use the 64 bit version.
mlir::func::FuncOp getLlvmMemmove(FirOpBuilder &builder);

/// Get the LLVM intrinsic for `memset`. Use the 64 bit version.
mlir::func::FuncOp getLlvmMemset(FirOpBuilder &builder);

/// Get the C standard library `realloc` function.
mlir::func::FuncOp getRealloc(FirOpBuilder &builder);

/// Get the `llvm.get.rounding` intrinsic.
mlir::func::FuncOp getLlvmGetRounding(FirOpBuilder &builder);

/// Get the `llvm.set.rounding` intrinsic.
mlir::func::FuncOp getLlvmSetRounding(FirOpBuilder &builder);

/// Get the `llvm.stacksave` intrinsic.
mlir::func::FuncOp getLlvmStackSave(FirOpBuilder &builder);

/// Get the `llvm.stackrestore` intrinsic.
mlir::func::FuncOp getLlvmStackRestore(FirOpBuilder &builder);

/// Get the `llvm.init.trampoline` intrinsic.
mlir::func::FuncOp getLlvmInitTrampoline(FirOpBuilder &builder);

/// Get the `llvm.adjust.trampoline` intrinsic.
mlir::func::FuncOp getLlvmAdjustTrampoline(FirOpBuilder &builder);

} // namespace fir::factory

#endif // FLANG_OPTIMIZER_BUILDER_LOWLEVELINTRINSICS_H
