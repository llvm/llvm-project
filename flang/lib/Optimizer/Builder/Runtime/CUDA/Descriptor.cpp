
//===-- Allocatable.cpp -- Allocatable statements lowering ----------------===//
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

#include "flang/Optimizer/Builder/Runtime/CUDA/Descriptor.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/CUDA/descriptor.h"

using namespace Fortran::runtime::cuda;

void fir::runtime::cuda::genSyncGlobalDescriptor(fir::FirOpBuilder &builder,
                                                 mlir::Location loc,
                                                 mlir::Value hostPtr) {
  mlir::func::FuncOp callee =
      fir::runtime::getRuntimeFunc<mkRTKey(CUFSyncGlobalDescriptor)>(loc,
                                                                     builder);
  auto fTy = callee.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
      builder, loc, fTy, hostPtr, sourceFile, sourceLine)};
  fir::CallOp::create(builder, loc, callee, args);
}

void fir::runtime::cuda::genDescriptorCheckSection(fir::FirOpBuilder &builder,
                                                   mlir::Location loc,
                                                   mlir::Value desc) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(CUFDescriptorCheckSection)>(loc,
                                                                       builder);
  auto fTy = func.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
  llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
      builder, loc, fTy, desc, sourceFile, sourceLine)};
  fir::CallOp::create(builder, loc, func, args);
}

void fir::runtime::cuda::genSetAllocatorIndex(fir::FirOpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value desc,
                                              mlir::Value index) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(CUFSetAllocatorIndex)>(loc, builder);
  auto fTy = func.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
  llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
      builder, loc, fTy, desc, index, sourceFile, sourceLine)};
  fir::CallOp::create(builder, loc, func, args);
}
