//===-- CUDAKernelOpInterface.td ----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Marker interface for operations that semantically represent a CUDA kernel
// boundary in Flang's lowered IR (e.g. cuf.kernel). The interface is defined
// in the FIR dialect so that FIR-level transformations can detect such
// boundaries without having to depend on the CUF dialect (which already
// depends on FIR; the reverse edge would form a circular library dependency
// in BUILD_SHARED_LIBS=ON builds).
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_CUDA_KERNEL_OP_INTERFACE
#define FORTRAN_DIALECT_CUDA_KERNEL_OP_INTERFACE

include "mlir/IR/OpBase.td"

def CUDAKernelOpInterface : OpInterface<"CUDAKernelOpInterface"> {
  let description = [{
    Marker interface for operations that act as a CUDA kernel boundary in
    Flang's lowered IR. Operations implementing this interface are treated by
    FIR transformations as opaque kernel regions whose host-visible captures
    must not be folded across the boundary.
  }];

  let cppNamespace = "::fir";
}

#endif // FORTRAN_DIALECT_CUDA_KERNEL_OP_INTERFACE
