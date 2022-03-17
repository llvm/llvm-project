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

/// Get the `llvm.stacksave` intrinsic.
mlir::func::FuncOp getLlvmStackSave(FirOpBuilder &builder);

/// Get the `llvm.stackrestore` intrinsic.
mlir::func::FuncOp getLlvmStackRestore(FirOpBuilder &builder);

} // namespace fir::factory

#endif // FLANG_OPTIMIZER_BUILDER_LOWLEVELINTRINSICS_H
