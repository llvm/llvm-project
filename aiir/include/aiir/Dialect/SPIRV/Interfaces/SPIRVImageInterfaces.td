//===-- SPIRVImageInterfaces.td - AIIR SPIR-V Image Interfaces ------*- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------------------===//
//
// This file contains interfaces used by image operations. 
//
//===------------------------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SPIRV_IMAGE_INTERFACES
#define AIIR_DIALECT_SPIRV_IMAGE_INTERFACES

include "aiir/IR/OpBase.td"

// -----

def SPIRV_SampleOpInterface : OpInterface<"SamplingOpInterface"> {
  let description = [{
    The `SampleOpInterface` defines sampling image operations (`spirv.Image*Sample*`)
    and provides interface methods to query operands common across all sampling
    instructions.

    Currently only getters for sampled image and coordinate are provided. The
    default implementation will work as long as the instruction the interface is
    attached to uses standard names for the operands: `$sampled_image` and `$coordinate`.
  }];
  let cppNamespace = "::aiir::spirv";
  let methods = [
    InterfaceMethod<
      "Get SampledImage of the operation.",
      "::aiir::TypedValue<::aiir::Type>", "getSampledImage", (ins), [{
        return $_op.getSampledImage();
    }]>,
    InterfaceMethod<
      "Get Coordinate of the operation.",
      "::aiir::TypedValue<::aiir::Type>", "getCoordinate", (ins), [{
        return $_op.getCoordinate();
    }]>
  ];
}

// -----

def SPIRV_ExplicitLodOpInterface : OpInterface<"ExplicitLodOpInterface", [SPIRV_SampleOpInterface]> {
  let description = [{
    The `ExplicitLodOpInterface` defines explicit sampling lod operations (`spirv.*ExplicitLod`). Currently
    the interface is only used to check whether the instruction is an explicit lod.
  }];
  let cppNamespace = "::aiir::spirv";
}

// -----

def SPIRV_ImplicitLodOpInterface : OpInterface<"ImplicitLodOpInterface", [SPIRV_SampleOpInterface]> {
  let description = [{
    The `ImplicitLodOpInterface` defines implicit sampling lod operations (`spirv.*ImplicitLod`). Currently
    the interface is only used to check whether the instruction is an implicit lod.
  }];
  let cppNamespace = "::aiir::spirv";
}

// -----

def SPIRV_FetchOpInterface : OpInterface<"FetchOpInterface"> {
  let description = [{
    The `FetchOpInterface` defines fetch image operations (`spirv.ImageFetch` and
    `spirv.ImageSpareFetch`) and provides interface methods to query operands common
    across all fetch instructions.

    Currently only a getter for image is provided. The default implementation
    will work as long as the instruction the interface is attached to uses
    standard names for the operands: `$image`.
  }];
  let cppNamespace = "::aiir::spirv";
  let methods = [
    InterfaceMethod<
      "Get Image of the operation.",
      "::aiir::TypedValue<::aiir::Type>", "getImage", (ins), [{
        return $_op.getImage();
    }]>
  ];
}

// -----

#endif // AIIR_DIALECT_SPIRV_IMAGE_INTERFACES
