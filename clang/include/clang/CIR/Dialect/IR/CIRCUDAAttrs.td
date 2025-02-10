//===---- CIRCUDAAttrs.td - CIR dialect attrs for CUDA -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect attributes for OpenCL.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_DIALECT_CIR_CUDA_ATTRS
#define MLIR_CIR_DIALECT_CIR_CUDA_ATTRS

//===----------------------------------------------------------------------===//
// CUDAKernelNameAttr
//===----------------------------------------------------------------------===//

def CUDAKernelNameAttr : CIR_Attr<"CUDAKernelName",
                                  "cuda_kernel_name"> {
  let summary = "Device-side function name for this stub.";
  let description =
  [{
    This attribute is attached to function definitions and records the
    mangled name of the kernel function used on the device.

    In CUDA, global functions (kernels) are processed differently for host
    and device. On host, Clang generates device stubs; on device, they are
    treated as normal functions. As they probably have different mangled
    names, we must record the corresponding device-side name for a stub.
  }];

  let parameters = (ins "std::string":$kernel_name);
  let assemblyFormat = "`<` $kernel_name `>`";
}

#endif // MLIR_CIR_DIALECT_CIR_CUDA_ATTRS
