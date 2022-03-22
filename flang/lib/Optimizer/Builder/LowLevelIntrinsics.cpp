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

mlir::FuncOp fir::factory::getLlvmMemcpy(fir::FirOpBuilder &builder) {
  auto ptrTy = builder.getRefType(builder.getIntegerType(8));
  llvm::SmallVector<mlir::Type> args = {ptrTy, ptrTy, builder.getI64Type(),
                                        builder.getI1Type()};
  auto memcpyTy =
      mlir::FunctionType::get(builder.getContext(), args, llvm::None);
  return builder.addNamedFunction(builder.getUnknownLoc(),
                                  "llvm.memcpy.p0i8.p0i8.i64", memcpyTy);
}

mlir::FuncOp fir::factory::getLlvmMemmove(fir::FirOpBuilder &builder) {
  auto ptrTy = builder.getRefType(builder.getIntegerType(8));
  llvm::SmallVector<mlir::Type> args = {ptrTy, ptrTy, builder.getI64Type(),
                                        builder.getI1Type()};
  auto memmoveTy =
      mlir::FunctionType::get(builder.getContext(), args, llvm::None);
  return builder.addNamedFunction(builder.getUnknownLoc(),
                                  "llvm.memmove.p0i8.p0i8.i64", memmoveTy);
}

mlir::FuncOp fir::factory::getLlvmMemset(fir::FirOpBuilder &builder) {
  auto ptrTy = builder.getRefType(builder.getIntegerType(8));
  llvm::SmallVector<mlir::Type> args = {ptrTy, ptrTy, builder.getI64Type(),
                                        builder.getI1Type()};
  auto memsetTy =
      mlir::FunctionType::get(builder.getContext(), args, llvm::None);
  return builder.addNamedFunction(builder.getUnknownLoc(),
                                  "llvm.memset.p0i8.p0i8.i64", memsetTy);
}

mlir::FuncOp fir::factory::getRealloc(fir::FirOpBuilder &builder) {
  auto ptrTy = builder.getRefType(builder.getIntegerType(8));
  llvm::SmallVector<mlir::Type> args = {ptrTy, builder.getI64Type()};
  auto reallocTy = mlir::FunctionType::get(builder.getContext(), args, {ptrTy});
  return builder.addNamedFunction(builder.getUnknownLoc(), "realloc",
                                  reallocTy);
}

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

mlir::FuncOp fir::factory::getLlvmInitTrampoline(fir::FirOpBuilder &builder) {
  auto ptrTy = builder.getRefType(builder.getIntegerType(8));
  auto funcTy = mlir::FunctionType::get(builder.getContext(),
                                        {ptrTy, ptrTy, ptrTy}, llvm::None);
  return builder.addNamedFunction(builder.getUnknownLoc(),
                                  "llvm.init.trampoline", funcTy);
}

mlir::FuncOp fir::factory::getLlvmAdjustTrampoline(fir::FirOpBuilder &builder) {
  auto ptrTy = builder.getRefType(builder.getIntegerType(8));
  auto funcTy = mlir::FunctionType::get(builder.getContext(), {ptrTy}, {ptrTy});
  return builder.addNamedFunction(builder.getUnknownLoc(),
                                  "llvm.adjust.trampoline", funcTy);
}
