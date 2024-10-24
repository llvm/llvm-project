//===-- Lower/Cuda.h -- Cuda Fortran utilities ------------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_CUDA_H
#define FORTRAN_LOWER_CUDA_H

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"

namespace Fortran::lower {
// Check if the insertion point is currently in a device context. HostDevice
// subprogram are not considered fully device context so it will return false
// for it.
// If the insertion point is inside an OpenACC region op, it is considered
// device context.
static bool isCudaDeviceContext(fir::FirOpBuilder &builder) {
  if (builder.getRegion().getParentOfType<cuf::KernelOp>())
    return true;
  if (builder.getRegion()
          .getParentOfType<mlir::acc::ComputeRegionOpInterface>())
    return true;
  if (auto funcOp = builder.getRegion().getParentOfType<mlir::func::FuncOp>()) {
    if (auto cudaProcAttr =
            funcOp.getOperation()->getAttrOfType<cuf::ProcAttributeAttr>(
                cuf::getProcAttrName())) {
      return cudaProcAttr.getValue() != cuf::ProcAttribute::Host &&
             cudaProcAttr.getValue() != cuf::ProcAttribute::HostDevice;
    }
  }
  return false;
}
} // end namespace Fortran::lower

#endif // FORTRAN_LOWER_CUDA_H
