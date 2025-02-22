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
                                  "cu.kernel_name"> {
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

def CUDABinaryHandleAttr : CIR_Attr<"CUDABinaryHandle",
                                  "cu.binary_handle"> {
  let summary = "Fat binary handle for device code.";
  let description =
  [{
    This attribute is attached to the ModuleOp and records the binary file
    name passed to host.

    CUDA first compiles device-side code into a fat binary file. The file
    name is then passed into host-side code, which is used to create a handle
    and then generate various registration functions.
  }];

  let parameters = (ins "std::string":$name);
  let assemblyFormat = "`<` $name `>`";
}

#endif // MLIR_CIR_DIALECT_CIR_CUDA_ATTRS
