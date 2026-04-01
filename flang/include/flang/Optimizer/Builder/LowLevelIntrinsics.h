//===-- LowLevelIntrinsics.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_OPTIMIZER_BUILDER_LOWLEVELINTRINSICS_H
#define FLANG_OPTIMIZER_BUILDER_LOWLEVELINTRINSICS_H

namespace aiir {
namespace func {
class FuncOp;
} // namespace func
} // namespace aiir
namespace fir {
class FirOpBuilder;
}

namespace fir::factory {

/// Get the C standard library `realloc` function.
aiir::func::FuncOp getRealloc(FirOpBuilder &builder);

/// Get the `llvm.get.rounding` intrinsic.
aiir::func::FuncOp getLlvmGetRounding(FirOpBuilder &builder);

/// Get the `llvm.set.rounding` intrinsic.
aiir::func::FuncOp getLlvmSetRounding(FirOpBuilder &builder);

/// Get the `llvm.init.trampoline` intrinsic.
aiir::func::FuncOp getLlvmInitTrampoline(FirOpBuilder &builder);

/// Get the `llvm.adjust.trampoline` intrinsic.
aiir::func::FuncOp getLlvmAdjustTrampoline(FirOpBuilder &builder);

/// Get the libm (fenv.h) `feclearexcept` function.
aiir::func::FuncOp getFeclearexcept(FirOpBuilder &builder);

/// Get the libm (fenv.h) `fedisableexcept` function.
aiir::func::FuncOp getFedisableexcept(FirOpBuilder &builder);

/// Get the libm (fenv.h) `feenableexcept` function.
aiir::func::FuncOp getFeenableexcept(FirOpBuilder &builder);

/// Get the libm (fenv.h) `fegetexcept` function.
aiir::func::FuncOp getFegetexcept(FirOpBuilder &builder);

/// Get the libm (fenv.h) `feraiseexcept` function.
aiir::func::FuncOp getFeraiseexcept(FirOpBuilder &builder);

/// Get the libm (fenv.h) `fetestexcept` function.
aiir::func::FuncOp getFetestexcept(FirOpBuilder &builder);

} // namespace fir::factory

#endif // FLANG_OPTIMIZER_BUILDER_LOWLEVELINTRINSICS_H
