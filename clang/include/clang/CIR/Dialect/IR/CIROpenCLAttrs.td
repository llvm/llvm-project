//===- CIROpenCLAttrs.td - CIR dialect attrs for OpenCL ----*- tablegen -*-===//
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

#ifndef MLIR_CIR_DIALECT_CIR_OPENCL_ATTRS
#define MLIR_CIR_DIALECT_CIR_OPENCL_ATTRS

//===----------------------------------------------------------------------===//
// OpenCLKernelMetadataAttr
//===----------------------------------------------------------------------===//

def OpenCLKernelMetadataAttr
    : CIR_Attr<"OpenCLKernelMetadata", "cl.kernel_metadata"> {
  
  let summary = "OpenCL kernel metadata";
  let description = [{
    Provide the required information of an OpenCL kernel for the SPIR-V backend.

    The `work_group_size_hint` and `reqd_work_group_size` parameter are integer
    arrays with 3 elements that provide hints for the work-group size and the
    required work-group size, respectively.

    The `vec_type_hint` parameter is a type attribute that provides a hint for
    the vectorization. It can be a CIR or LLVM type, depending on the lowering
    stage.

    The `vec_type_hint_signedness` parameter is a boolean that indicates the
    signedness of the vector type hint. It's useful when LLVM type is set in
    `vec_type_hint`, which is signless by design. It should be set if and only
    if the `vec_type_hint` is present.

    The `intel_reqd_sub_group_size` parameter is an integer that restricts the
    sub-group size to the specified value.

    Example:
    ```
    #fn_attr = #cir<extra({cl.kernel_metadata = #cir.cl.kernel_metadata<
      work_group_size_hint = [8 : i32, 16 : i32, 32 : i32],
      reqd_work_group_size = [1 : i32, 2 : i32, 4 : i32],
      vec_type_hint = !s32i,
      vec_type_hint_signedness = 1,
      intel_reqd_sub_group_size = 8 : i32
    >})>

    cir.func @kernel(%arg0: !s32i) extra(#fn_attr) {
      cir.return
    }
    ```
  }];

  let parameters = (ins
    OptionalParameter<"ArrayAttr">:$work_group_size_hint,
    OptionalParameter<"ArrayAttr">:$reqd_work_group_size,
    OptionalParameter<"TypeAttr">:$vec_type_hint,
    OptionalParameter<"std::optional<bool>">:$vec_type_hint_signedness,
    OptionalParameter<"IntegerAttr">:$intel_reqd_sub_group_size
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Extract the signedness from int or int vector types.
    static std::optional<bool> isSignedHint(mlir::Type vecTypeHint);
  }];

  let extraClassDefinition = [{
    std::optional<bool> $cppClass::isSignedHint(mlir::Type hintQTy) {
      // Only types in CIR carry signedness
      if (!mlir::isa<mlir::cir::CIRDialect>(hintQTy.getDialect()))
        return std::nullopt;
      
      // See also clang::CodeGen::CodeGenFunction::EmitKernelMetadata
      auto hintEltQTy = mlir::dyn_cast<mlir::cir::VectorType>(hintQTy);
      auto isCIRSignedIntType = [](mlir::Type t) {
        return mlir::isa<mlir::cir::IntType>(t) &&
               mlir::cast<mlir::cir::IntType>(t).isSigned();
      };
      return isCIRSignedIntType(hintQTy) ||
              (hintEltQTy && isCIRSignedIntType(hintEltQTy.getEltType()));
    }
  }];

}

//===----------------------------------------------------------------------===//
// OpenCLKernelArgMetadataAttr
//===----------------------------------------------------------------------===//

def OpenCLKernelArgMetadataAttr
    : CIR_Attr<"OpenCLKernelArgMetadata", "cl.kernel_arg_metadata"> {
  
  let summary = "OpenCL kernel argument metadata";
  let description = [{
    Provide the required information of an OpenCL kernel argument for the SPIR-V
    backend.

    All parameters are arrays, containing the information of the argument in
    the same order as they appear in the source code.

    The `addr_space` parameter is an array of I32 that provides the address
    space of the argument. It's useful for special types like `image`, which
    have implicit global address space.

    Other parameters are arrays of strings that pass through the information
    from the source code correspondingly.

    All the fields are mandatory except for `name`, which is optional.

    Example:
    ```
    #fn_attr = #cir<extra({cl.kernel_arg_metadata = #cir.cl.kernel_arg_metadata<
      addr_space = [1 : i32],
      access_qual = ["none"],
      type = ["char*"],
      base_type = ["char*"],
      type_qual = [""],
      name = ["in"]
    >})>

    cir.func @kernel(%arg0: !s32i) extra(#fn_attr) {
      cir.return
    }
    ```
  }];

  let parameters = (ins
    "ArrayAttr":$addr_space,
    "ArrayAttr":$access_qual,
    "ArrayAttr":$type,
    "ArrayAttr":$base_type,
    "ArrayAttr":$type_qual,
    OptionalParameter<"ArrayAttr">:$name
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// OpenCLVersionAttr
//===----------------------------------------------------------------------===//

def OpenCLVersionAttr : CIR_Attr<"OpenCLVersion", "cl.version"> {
  let summary = "OpenCL version";
  let parameters = (ins "int32_t":$major_version, "int32_t":$minor_version);
  let description = [{
    Represents the version of OpenCL.

    Example:
    ```
    // Module compiled from OpenCL 1.2.
    module attributes {cir.cl.version = cir.cl.version<1, 2>} {}
    // Module compiled from OpenCL 3.0.
    module attributes {cir.cl.version = cir.cl.version<3, 0>} {}
    ```
  }];
  let assemblyFormat = "`<` $major_version `,` $minor_version `>`";
}


//===----------------------------------------------------------------------===//
// OpenCLKernelAttr
//===----------------------------------------------------------------------===//

// TODO: It might be worthwhile to introduce a generic attribute applicable to
// all offloading languages.
def OpenCLKernelAttr : CIRUnitAttr<
    "OpenCLKernel", "cl.kernel"> {
  let summary = "OpenCL kernel";
  let description = [{
    Indicate the function is a OpenCL kernel.
  }];

  let storageType = [{ OpenCLKernelAttr }];
}

#endif // MLIR_CIR_DIALECT_CIR_OPENCL_ATTRS
