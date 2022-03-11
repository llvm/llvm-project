//===-- LowLevelIntrinsics.cpp --------------------------------------------===//
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
//
// Low level intrinsic functions.
//
// These include LLVM intrinsic calls and standard C library calls.
// Target-specific calls, such as OS functions, should be factored in other
// file(s).
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"

mlir::FuncOp fir::factory::getLlvmStackSave(fir::FirOpBuilder &builder) {
  auto ptrTy = builder.getRefType(builder.getIntegerType(8));
  auto funcTy =
      mlir::FunctionType::get(builder.getContext(), llvm::None, {ptrTy});
  return builder.addNamedFunction(builder.getUnknownLoc(), "llvm.stacksave",
                                  funcTy);
}

mlir::FuncOp fir::factory::getLlvmStackRestore(fir::FirOpBuilder &builder) {
  auto ptrTy = builder.getRefType(builder.getIntegerType(8));
  auto funcTy =
      mlir::FunctionType::get(builder.getContext(), {ptrTy}, llvm::None);
  return builder.addNamedFunction(builder.getUnknownLoc(), "llvm.stackrestore",
                                  funcTy);
}
