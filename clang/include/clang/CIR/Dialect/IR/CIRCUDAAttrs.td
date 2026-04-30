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
  let canHaveIllegalCXXABIType = 0;
}

def CIR_CUDAExternallyInitializedAttr : CIR_Attr<"CUDAExternallyInitialized",
                                             "cu.externally_initialized"> {
  let summary = "The marked variable is externally initialized.";
  let description =
  [{
    CUDA __device__ and __constant__ variables, along with surface and
    textures, might be initialized by host, hence "externally initialized".
    Therefore they must be emitted even if they are not referenced.

    The attribute corresponds to the attribute on LLVM with the same name.
  }];
  let canHaveIllegalCXXABIType = 0;
}

// Enum for device variable kinds
def CIR_CUDADeviceVarKind : I32EnumAttr<"CUDADeviceVarKind",
    "CUDA device variable kind", [
  I32EnumAttrCase<"Variable", 0>,
  I32EnumAttrCase<"Surface", 1>,   // Future
  I32EnumAttrCase<"Texture", 2>,   // Future
]> {
  let cppNamespace = "::cir";
}

// Attribute carrying device variable registration flags
def CIR_CUDAVarRegistrationInfoAttr : CIR_Attr<"CUDAVarRegistrationInfo", "cu.var_registration"> {
  let summary = "Device variable registration flags.";
  let parameters = (ins
    "std::string":$device_side_name,
    "CUDADeviceVarKind":$kind,
    // Note: isExtern is only exercisable once HIP managed variable support
    // lands. The only path to isExtern=true requires an extern __managed__
    // variable, which bypasses the external storage filter in
    // handleVarRegistration via hasAttr<HIPManagedAttr>().
    "bool":$isExtern,
    "bool":$isConstant,
    "bool":$isManaged
  );

  let hasCustomAssemblyFormat = 1;
}

#endif // CLANG_CIR_DIALECT_IR_CIRCUDAATTRS_TD
