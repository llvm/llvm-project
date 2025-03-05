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

def CUDAShadowNameAttr : CIR_Attr<"CUDAShadowName",
                                  "cu.shadow_name"> {
  let summary = "Device-side global variable name for this shadow.";
  let description =
  [{
    This attribute is attached to global variable definitions and records the
    mangled name of the global variable used on the device.

    In CUDA, __device__, __constant__ and __shared__ variables, as well as 
    surface and texture variables, will generate a shadow symbol on host.
    We must preserve the correspodence in order to generate registration
    functions.
  }];

  let parameters = (ins "std::string":$device_side_name);
  let assemblyFormat = "`<` $device_side_name `>`";
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

def CUDAExternallyInitializedAttr : CIR_Attr<"CUDAExternallyInitialized",
                                             "cu.externally_initialized"> {
  let summary = "The marked variable is externally initialized.";
  let description =
  [{
    CUDA __device__ and __constant__ variables, along with surface and
    textures, might be initialized by host, hence "externally initialized".
    Therefore they must be emitted even if they are not referenced.

    The attribute corresponds to the attribute on LLVM with the same name.
  }];
}

#endif // MLIR_CIR_DIALECT_CIR_CUDA_ATTRS
