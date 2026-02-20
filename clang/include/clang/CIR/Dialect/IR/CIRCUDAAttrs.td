//===---- CIRCUDAAttrs.td - CIR dialect attrs for CUDA -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect attributes for CUDA.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRCUDAATTRS_TD
#define CLANG_CIR_DIALECT_IR_CIRCUDAATTRS_TD

//===----------------------------------------------------------------------===//
// CUDAKernelNameAttr
//===----------------------------------------------------------------------===//

def CIR_CUDAKernelNameAttr : CIR_Attr<"CUDAKernelName", "cu.kernel_name"> {
  let summary = "Device-side function name for this stub.";
  let description =
  [{
    This attribute is attached to function definitions and records the
    mangled name of the kernel function used on the device.

    In CUDA, global functions (kernels) are processed differently for host
    and device. On host, Clang generates device stubs; on device, they are
    treated as normal functions. As they probably have different mangled
    names, we must record the corresponding device-side name for a stub.
    Preserving the device-side kernel name is crucial for performing its
    respective function runtime registration on the host.
  }];

  let parameters = (ins "std::string":$kernel_name);
  let assemblyFormat = "`<` $kernel_name `>`";
}


#endif // CLANG_CIR_DIALECT_IR_CIRCUDAATTRS_TD