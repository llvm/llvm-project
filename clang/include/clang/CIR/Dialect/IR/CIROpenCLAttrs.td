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

#ifndef CLANG_CIR_DIALECT_IR_CIROPENCLATTRS_TD
#define CLANG_CIR_DIALECT_IR_CIROPENCLATTRS_TD

//===----------------------------------------------------------------------===//
// OpenCLKernelArgMetadataAttr
//===----------------------------------------------------------------------===//

def CIR_OpenCLKernelArgMetadataAttr
    : CIR_Attr<"OpenCLKernelArgMetadata", "cl.kernel_arg_metadata"> {
  let summary = "OpenCL kernel argument metadata";
  let description = [{
    Stores the OpenCL kernel argument metadata emitted to LLVM IR as
    `kernel_arg_*` metadata.

    All parameters are arrays containing the argument information in source
    order. The `name` field is optional and is emitted only when requested by
    `-cl-kernel-arg-info`.
  }];

  let parameters = (ins
    CIR_LangAddressSpaceArrayAttr:$addr_space,
    CIR_StringArrayAttr:$access_qual,
    CIR_StringArrayAttr:$type,
    CIR_StringArrayAttr:$base_type,
    CIR_StringArrayAttr:$type_qual,
    CIR_StringArrayAttrOrNull:$name
  );

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
  let canHaveIllegalCXXABIType = 0;
}

#endif // CLANG_CIR_DIALECT_IR_CIROPENCLATTRS_TD
