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

static inline unsigned getAllocatorIdx(const Fortran::semantics::Symbol &sym) {
  std::optional<Fortran::common::CUDADataAttr> cudaAttr =
      Fortran::semantics::GetCUDADataAttr(&sym.GetUltimate());
  if (cudaAttr) {
    if (*cudaAttr == Fortran::common::CUDADataAttr::Pinned)
      return kPinnedAllocatorPos;
    if (*cudaAttr == Fortran::common::CUDADataAttr::Device)
      return kDeviceAllocatorPos;
    if (*cudaAttr == Fortran::common::CUDADataAttr::Managed)
      return kManagedAllocatorPos;
    if (*cudaAttr == Fortran::common::CUDADataAttr::Unified)
      return kUnifiedAllocatorPos;
  }
  return kDefaultAllocator;
}

} // end namespace Fortran::lower

#endif // FORTRAN_LOWER_CUDA_H
