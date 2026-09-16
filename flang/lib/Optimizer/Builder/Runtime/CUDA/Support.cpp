//===-- Support.cpp -- Lowering helper for CUDA runtime functions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/CUDA/Support.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/CUDA/allocatable.h"

static constexpr llvm::StringRef kCudaDeviceSynchronizeName =
    "_QPcudadevicesynchronize";

using namespace Fortran::runtime::cuda;

void fir::runtime::cuda::genCUDADeviceSynchronize(fir::FirOpBuilder &builder,
                                                  mlir::Location loc) {
  mlir::func::FuncOp func =
      builder.getNamedFunction(kCudaDeviceSynchronizeName);
  if (!func) {
    mlir::FunctionType funcType = mlir::FunctionType::get(
        builder.getContext(), {}, {builder.getI32Type()});
    func = builder.createFunction(loc, kCudaDeviceSynchronizeName, funcType);
    func->setAttr(
        fir::getFortranProcedureFlagsAttrName(),
        fir::FortranProcedureFlagsEnumAttr::get(
            builder.getContext(), fir::FortranProcedureFlagsEnum::intrinsic));
    func.setPrivate();
  }
  auto call = fir::CallOp::create(builder, loc, func, mlir::ValueRange{});
  call.setProcedureAttrsAttr(fir::FortranProcedureFlagsEnumAttr::get(
      builder.getContext(), fir::FortranProcedureFlagsEnum::intrinsic));
}

mlir::Value fir::runtime::cuda::genDeviceIsActive(fir::FirOpBuilder &builder,
                                                  mlir::Location loc) {
  mlir::func::FuncOp func =
      fir::runtime::getRuntimeFunc<mkRTKey(CUFDeviceIsActive)>(loc, builder);
  auto call = fir::CallOp::create(builder, loc, func, mlir::ValueRange{});
  return builder.createConvert(loc, builder.getI1Type(), call.getResult(0));
}
