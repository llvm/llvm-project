//===- CUFAttr.td - CUF Attributes -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CUF dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_CUF_CUFATTRS
#define FORTRAN_DIALECT_CUF_CUFATTRS

include "flang/Optimizer/Dialect/CUF/CUFDialect.td"
include "mlir/IR/EnumAttr.td"

class cuf_Attr<string name> : AttrDef<CUFDialect, name>;

def cuf_DataAttribute : I32EnumAttr<
    "DataAttribute",
    "CUDA Fortran variable attributes",
    [
      I32EnumAttrCase<"Constant", 0, "constant">,
      I32EnumAttrCase<"Device", 1, "device">,
      I32EnumAttrCase<"Managed", 2, "managed">,
      I32EnumAttrCase<"Pinned", 3, "pinned">,
      I32EnumAttrCase<"Shared", 4, "shared">,
      I32EnumAttrCase<"Unified", 5, "unified">,
      // Texture is omitted since it is obsolete and rejected by semantic.
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::cuf";
}

def cuf_DataAttributeAttr :
    EnumAttr<CUFDialect, cuf_DataAttribute, "cuda"> {
  let assemblyFormat = [{ ```<` $value `>` }];
}

def cuf_ProcAttribute : I32EnumAttr<
    "ProcAttribute", "CUDA Fortran procedure attributes",
    [
      I32EnumAttrCase<"Host", 0, "host">,
      I32EnumAttrCase<"Device", 1, "device">,
      I32EnumAttrCase<"HostDevice", 2, "host_device">,
      I32EnumAttrCase<"Global", 3, "global">,
      I32EnumAttrCase<"GridGlobal", 4, "grid_global">,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::cuf";
}

def cuf_ProcAttributeAttr :
    EnumAttr<CUFDialect, cuf_ProcAttribute, "cuda_proc"> {
  let assemblyFormat = [{ ```<` $value `>` }];
}

def cuf_LaunchBoundsAttr : cuf_Attr<"LaunchBounds"> {
  let mnemonic = "launch_bounds";

  let parameters = (ins
    "mlir::IntegerAttr":$maxTPB,
    "mlir::IntegerAttr":$minBPM,
    OptionalParameter<"mlir::IntegerAttr">:$upperBoundClusterSize
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def cuf_ClusterDimsAttr : cuf_Attr<"ClusterDims"> {
  let mnemonic = "cluster_dims";

  let parameters = (ins
    "mlir::IntegerAttr":$x,
    "mlir::IntegerAttr":$y,
    "mlir::IntegerAttr":$z
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def cuf_DataTransferKind : I32EnumAttr<
    "DataTransferKind", "CUDA Fortran data transfer kind",
    [
      I32EnumAttrCase<"DeviceHost", 0, "device_host">,
      I32EnumAttrCase<"HostDevice", 1, "host_device">,
      I32EnumAttrCase<"DeviceDevice", 2, "device_device">,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::cuf";
}

def cuf_DataTransferKindAttr :
    EnumAttr<CUFDialect, cuf_DataTransferKind, "cuda_transfer"> {
  let assemblyFormat = [{ ```<` $value `>` }];
}

#endif // FORTRAN_DIALECT_CUF_CUFATTRS
