//===- CIRTBAAAttrs.td - CIR dialect TBAA attributes -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CIR dialect attributes describing type-based alias
// analysis (TBAA) information.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRTBAAATTRS_TD
#define CLANG_CIR_DIALECT_IR_CIRTBAAATTRS_TD

//===----------------------------------------------------------------------===//
// TBAAOmnipotentCharAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAOmnipotentCharAttr
    : CIR_Attr<"TBAAOmnipotentChar", "tbaa_omnipotent_char"> {
  let summary = "TBAA type descriptor for the omnipotent char type";
  let description = [{
    Describes the character types, which may alias any other type. It is the
    parent of every other TBAA type descriptor. The TBAA root it hangs from is
    implied by the source language of the module and is materialized when
    lowering to LLVM.

    Example:
    ```mlir
    #tbaa_char = #cir.tbaa_omnipotent_char
    ```
  }];
  let canHaveIllegalCXXABIType = 0;
}

//===----------------------------------------------------------------------===//
// TBAAScalarAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAScalarAttr : CIR_Attr<"TBAAScalar", "tbaa_scalar"> {
  let summary = "TBAA type descriptor for a scalar type";
  let description = [{
    Describes a scalar type named `id` whose objects are `size` bytes wide.
    The optional `parent` must be another scalar type descriptor; when absent,
    the parent is the omnipotent char type.

    The descriptor intentionally does not refer to a CIR type: it describes the
    source-level type of the access, which must survive passes that rewrite
    the CIR types of the memory operations carrying it.

    Example:
    ```mlir
    #tbaa_int = #cir.tbaa_scalar<id = "int", size = 4>
    #tbaa_any_ptr = #cir.tbaa_scalar<id = "any pointer", size = 8>
    #tbaa_p1_int = #cir.tbaa_scalar<id = "p1 int", size = 8,
                                    parent = #tbaa_any_ptr>
    ```
  }];

  let parameters = (ins StringRefParameter<>:$id, "uint64_t":$size,
                        OptionalParameter<"cir::TBAAScalarAttr">:$parent);

  let builders = [
    AttrBuilder<(ins "llvm::StringRef":$id, "uint64_t":$size), [{
      return $_get($_ctxt, id, size, /*parent=*/cir::TBAAScalarAttr());
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";
  let canHaveIllegalCXXABIType = 0;
}

//===----------------------------------------------------------------------===//
// TBAAMemberAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAMemberAttr : CIR_Attr<"TBAAMember", "tbaa_member"> {
  let summary = "A member of a TBAA struct type descriptor";
  let description = [{
    Describes a member of a struct type descriptor: the type descriptor of the
    member and its byte offset within the enclosing struct.
  }];

  let parameters = (ins "mlir::Attribute":$type_desc, "uint64_t":$offset);
  let assemblyFormat = "`<` $type_desc `,` $offset `>`";
  let genVerifyDecl = 1;
  let canHaveIllegalCXXABIType = 0;
}

//===----------------------------------------------------------------------===//
// TBAAStructAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAStructAttr : CIR_Attr<"TBAAStruct", "tbaa_struct"> {
  let summary = "TBAA type descriptor for a struct type";
  let description = [{
    Describes a struct type named `id` as a list of members, each with the
    type descriptor of the member and its byte offset. Struct type descriptors
    are used as the base type of struct-path access tags.

    Example:
    ```mlir
    #tbaa_int = #cir.tbaa_scalar<id = "int", size = 4>
    #tbaa_long = #cir.tbaa_scalar<id = "long", size = 8>
    #tbaa_S = #cir.tbaa_struct<"_ZTS1S", [<#tbaa_int, 0>, <#tbaa_long, 8>]>
    ```
  }];

  let parameters = (ins StringRefParameter<>:$id,
                        ArrayRefParameter<"cir::TBAAMemberAttr">:$members);
  let hasCustomAssemblyFormat = 1;
  let canHaveIllegalCXXABIType = 0;
}

//===----------------------------------------------------------------------===//
// TBAATagAttr
//===----------------------------------------------------------------------===//

def CIR_TBAATagAttr : CIR_Attr<"TBAATag", "tbaa_tag"> {
  let summary = "TBAA access tag of a memory operation";
  let description = [{
    Describes a memory access: an access of type `access` at byte `offset`
    within an object of type `base`. For accesses that are not struct member
    accesses, `base` and `access` are the same scalar type descriptor and
    `offset` is zero.

    Passes that change the width or the addressed object of a memory operation
    must drop its access tag: a missing tag is conservatively correct, a stale
    one is a miscompile.

    Example:
    ```mlir
    #tbaa_int = #cir.tbaa_scalar<id = "int", size = 4>
    #tbaa_S = #cir.tbaa_struct<"_ZTS1S", [<#tbaa_int, 0>, <#tbaa_int, 4>]>
    // Access to `int`.
    #tag0 = #cir.tbaa_tag<base = #tbaa_int, access = #tbaa_int, offset = 0>
    // Access to `S::y`.
    #tag1 = #cir.tbaa_tag<base = #tbaa_S, access = #tbaa_int, offset = 4>
    ```
  }];

  let parameters = (ins "mlir::Attribute":$base, "mlir::Attribute":$access,
                        "uint64_t":$offset);

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Attribute":$access), [{
      return $_get(access.getContext(), access, access, /*offset=*/0);
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
  let canHaveIllegalCXXABIType = 0;
}

#endif // CLANG_CIR_DIALECT_IR_CIRTBAAATTRS_TD
