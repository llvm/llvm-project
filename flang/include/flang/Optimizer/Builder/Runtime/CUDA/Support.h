//===-- Support.h - CUDA support runtime functions --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_CUDA_SUPPORT_H_
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_CUDA_SUPPORT_H_

namespace mlir {
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime::cuda {

/// Generate runtime call to synchronize the CUDA device.
void genCUDADeviceSynchronize(fir::FirOpBuilder &builder, mlir::Location loc);

} // namespace fir::runtime::cuda

#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_CUDA_SUPPORT_H_
