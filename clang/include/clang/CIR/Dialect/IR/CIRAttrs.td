//===- CIRAttrs.td - CIR dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_DIALECT_CIR_ATTRS
#define MLIR_CIR_DIALECT_CIR_ATTRS

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "clang/CIR/Dialect/IR/CIRDialect.td"

//===----------------------------------------------------------------------===//
// CIR Attrs
//===----------------------------------------------------------------------===//

class CIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// LangAttr
//===----------------------------------------------------------------------===//

def C : I32EnumAttrCase<"C", 1, "c">;
def CXX : I32EnumAttrCase<"CXX", 2, "cxx">;

def SourceLanguage : I32EnumAttr<"SourceLanguage", "Source language", [
  C, CXX
]> {
  let cppNamespace = "::mlir::cir";
  let genSpecializedAttr = 0;
}

def LangAttr : CIR_Attr<"Lang", "lang"> {
  let summary = "Module source language";
  let parameters = (ins SourceLanguage:$lang);
  let description = [{
    Represents the source language used to generate the module.

    Example:
    ```
    // Module compiled from C.
    module attributes {cir.lang = cir.lang<c>} {}
    // Module compiled from C++.
    module attributes {cir.lang = cir.lang<cxx>} {}
    ```
  }];
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    bool isC() const { return getLang() == SourceLanguage::C; };
    bool isCXX() const { return getLang() == SourceLanguage::CXX; };
  }];
}

//===----------------------------------------------------------------------===//
// NullAttr
//===----------------------------------------------------------------------===//

def NullAttr : CIR_Attr<"Null", "null", [TypedAttrInterface]> {
  let summary = "A simple attr to represent nullptr";
  let description = [{
    The NullAttr represents the value of nullptr within cir.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type), [{
      return $_get(type.getContext(), type);
    }]>
  ];

  let assemblyFormat = [{}];
}

//===----------------------------------------------------------------------===//
// BoolAttr
//===----------------------------------------------------------------------===//

def CIR_BoolAttr : CIR_Attr<"Bool", "bool", [TypedAttrInterface]> {
  let summary = "Represent true/false for !cir.bool types";
  let description = [{
    The BoolAttr represents a 'true' or 'false' value.
  }];

  let parameters = (ins AttributeSelfTypeParameter<
                        "", "mlir::cir::BoolType">:$type,
                    "bool":$value);

  let assemblyFormat = [{
    `<` $value `>`
  }];
}

//===----------------------------------------------------------------------===//
// ZeroAttr
//===----------------------------------------------------------------------===//

def ZeroAttr : CIR_Attr<"Zero", "zero", [TypedAttrInterface]> {
  let summary = "Attribute to represent zero initialization";
  let description = [{
    The ZeroAttr is used to indicate zero initialization on structs.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = [{}];
}

//===----------------------------------------------------------------------===//
// ConstArrayAttr
//===----------------------------------------------------------------------===//

def ConstArrayAttr : CIR_Attr<"ConstArray", "const_array", [TypedAttrInterface]> {
  let summary = "A constant array from ArrayAttr or StringRefAttr";
  let description = [{
    An CIR array attribute is an array of literals of the specified attr types.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "Attribute":$elts);

  // Define a custom builder for the type; that removes the need to pass
  // in an MLIRContext instance, as it can be infered from the `type`.
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::cir::ArrayType":$type,
                                        "Attribute":$elts), [{
      return $_get(type.getContext(), type, elts);
    }]>
  ];

  // Printing and parsing available in CIRDialect.cpp
  let hasCustomAssemblyFormat = 1;

  // Enable verifier.
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// ConstStructAttr
//===----------------------------------------------------------------------===//

def ConstStructAttr : CIR_Attr<"ConstStruct", "const_struct",
                               [TypedAttrInterface]> {
  let summary = "Represents a constant struct";
  let description = [{
    Effectively supports "struct-like" constants. It's must be built from
    an `mlir::ArrayAttr `instance where each elements is a typed attribute
    (`mlir::TypedAttribute`).

    Example:
    ```
    cir.global external @rgb2 = #cir.const_struct<{0 : i8,
                                                   5 : i64, #cir.null : !cir.ptr<i8>
                                                  }> : !cir.struct<"", i8, i64, !cir.ptr<i8>>
    ```
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "ArrayAttr":$members);

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::cir::StructType":$type,
                                        "ArrayAttr":$members), [{
      return $_get(type.getContext(), type, members);
    }]>
  ];

  let assemblyFormat = [{
    `<` custom<StructMembers>($members) `>`
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// IntegerAttr
//===----------------------------------------------------------------------===//

def IntAttr : CIR_Attr<"Int", "int", [TypedAttrInterface]> {
  let summary = "An Attribute containing a integer value";
  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "APInt":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const APInt &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>,
    AttrBuilderWithInferredContext<(ins "Type":$type, "int64_t":$value), [{
      IntType intType = type.cast<IntType>();
      mlir::APInt apValue(intType.getWidth(), value, intType.isSigned());
      return $_get(intType.getContext(), intType, apValue);
    }]>,
  ];
  let extraClassDeclaration = [{
    int64_t getSInt() const { return getValue().getSExtValue(); }
    uint64_t getUInt() const { return getValue().getZExtValue(); }
    bool isNullValue() const { return getValue() == 0; }
  }];
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// SignedOverflowBehaviorAttr
//===----------------------------------------------------------------------===//

def SignedOverflowBehaviorAttr : AttrDef<CIR_Dialect, "SignedOverflowBehavior"> {
  let mnemonic = "signed_overflow_behavior";
  let parameters = (ins
    "sob::SignedOverflowBehavior":$behavior
  );
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// GlobalViewAttr
//===----------------------------------------------------------------------===//

def GlobalViewAttr : CIR_Attr<"GlobalView", "global_view", [TypedAttrInterface]> {
  let summary = "Provides constant access to a global address";
  let description = [{
    Get constant address of global `symbol` and optionally apply offsets to
    access existing subelements. It provides a way to access globals from other
    global and always produces a pointer.

    The type of the input symbol can be different from `#cir.global_view`
    output type, since a given view of the global might require a static
    cast for initializing other globals.

    A list of indices can be optionally passed and each element subsequently
    indexes underlying types. For `symbol` types like `!cir.array`
    and `!cir.struct`, it leads to the constant address of sub-elements, while
    for `!cir.ptr`, an offset is applied. The first index is relative to the
    original symbol type, not the produced one.

    Example:

    ```
      cir.global external @s = @".str2": !cir.ptr<i8>
      cir.global external @x = #cir.global_view<@s> : !cir.ptr<i8>

      cir.global external @rgb = #cir.const_array<[0 : i8, -23 : i8, 33 : i8] : !cir.array<i8 x 3>>
      cir.global external @elt_ptr = #cir.global_view<@rgb, [1]> : !cir.ptr<i8>
      cir.global external @table_of_ptrs = #cir.const_array<[#cir.global_view<@rgb, [1]> : !cir.ptr<i8>] : !cir.array<!cir.ptr<i8> x 1>>
    ```
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "FlatSymbolRefAttr":$symbol,
                        OptionalParameter<"ArrayAttr">:$indices);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "FlatSymbolRefAttr":$symbol,
                                        CArg<"ArrayAttr", "{}">:$indices), [{
      return $_get(type.getContext(), type, symbol, indices);
    }]>
  ];

  // let genVerifyDecl = 1;
  let assemblyFormat = [{
    `<`
      $symbol
      (`,` $indices^)?
    `>`
  }];
}

//===----------------------------------------------------------------------===//
// TypeInfoAttr
//===----------------------------------------------------------------------===//

def TypeInfoAttr : CIR_Attr<"TypeInfo", "typeinfo", [TypedAttrInterface]> {
  let summary = "Represents a typeinfo used for RTTI";
  let description = [{
    The typeinfo data for a given class is stored into an ArrayAttr. The
    layout is determined by the C++ ABI used (clang only implements
    itanium on CIRGen).

    The verifier enforces that the output type is always a `!cir.struct`,
    and that the ArrayAttr element types match the equivalent member type
    for the resulting struct.

    Example:

    ```
    cir.global "private" external @_ZTVN10__cxxabiv120__si_class_type_infoE : !cir.ptr<i32>

    cir.global external @type_info_B = #cir.typeinfo<<
      {#cir.global_view<@_ZTVN10__cxxabiv120__si_class_type_infoE, [2]> : !cir.ptr<i8>}
    >> : !cir.struct<"", !cir.ptr<i8>>
    ```
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "mlir::ArrayAttr":$data);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "mlir::ArrayAttr":$data), [{
      return $_get(type.getContext(), type, data);
    }]>
  ];

  // Checks struct element types should match the array for every equivalent
  // element type.
  let genVerifyDecl = 1;
  let assemblyFormat = [{
    `<` custom<StructMembers>($data) `>`
  }];
}

//===----------------------------------------------------------------------===//
// VTableAttr
//===----------------------------------------------------------------------===//

def VTableAttr : CIR_Attr<"VTable", "vtable", [TypedAttrInterface]> {
  let summary = "Represents a C++ vtable";
  let description = [{
    Wraps a #cir.const_struct containing vtable data.

    Example:
    ```
    cir.global linkonce_odr @_ZTV1B = #cir.vtable<<
        {#cir.const_array<[#cir.null : !cir.ptr<i8>,
         #cir.global_view<@_ZTI1B> : !cir.ptr<i8>,
         #cir.global_view<@_ZN1BD1Ev> : !cir.ptr<i8>,
         #cir.global_view<@_ZN1BD0Ev> : !cir.ptr<i8>,
         #cir.global_view<@_ZNK1A5quackEv> : !cir.ptr<i8>]>
         : !cir.array<!cir.ptr<i8> x 5>}>>
      : !cir.struct<"", !cir.array<!cir.ptr<i8> x 5>>
    ```
  }];

  // `vtable_data` is const struct with one element, containing an array of
  // vtable information.
  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "ArrayAttr":$vtable_data);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "ArrayAttr":$vtable_data), [{
      return $_get(type.getContext(), type, vtable_data);
    }]>
  ];

  let genVerifyDecl = 1;
  let assemblyFormat = [{
    `<` custom<StructMembers>($vtable_data) `>`
  }];
}

//===----------------------------------------------------------------------===//
// AST Wrappers
//===----------------------------------------------------------------------===//

class ASTDecl<string name, string prefix, list<Trait> traits = []>
  : CIR_Attr<!strconcat("AST", name), !strconcat(prefix, ".ast"), traits> {
  string clang_name = !strconcat("const clang::", name, " *");

  let summary = !strconcat("Wraps a '", clang_name, "' AST node.");
  let description = [{
    Operations optionally refer to this node, they could be available depending
    on the CIR lowering stage. Whether it's attached to the appropriated
    CIR operation is delegated to the operation verifier.

    This always implies a non-null AST reference (verified).
  }];
  let parameters = (ins clang_name:$astDecl);

  // Printing and parsing available in CIRDialect.cpp
  let hasCustomAssemblyFormat = 1;

  // Enable verifier.
  let genVerifyDecl = 1;
}

def ASTFunctionDeclAttr : ASTDecl<"FunctionDecl", "fndecl">;
def ASTVarDeclAttr : ASTDecl<"VarDecl", "vardecl">;
def ASTRecordDeclAttr : ASTDecl<"RecordDecl", "recdecl">;


//===----------------------------------------------------------------------===//
// ExtraFuncAttr
//===----------------------------------------------------------------------===//

def ExtraFuncAttr : CIR_Attr<"ExtraFuncAttributes", "extra"> {
  let summary = "Represents aggregated attributes for a function";
  let description = [{
    This is a wrapper of dictionary attrbiute that contains extra attributes of
    a function.
  }];

  let parameters = (ins "DictionaryAttr":$elements);

  let assemblyFormat = [{ `(` $elements `)` }];

  // Printing and parsing also available in CIRDialect.cpp
}


def NoInline : I32EnumAttrCase<"NoInline", 1, "no">;
def AlwaysInline : I32EnumAttrCase<"AlwaysInline", 2, "always">;
def InlineHint : I32EnumAttrCase<"InlineHint", 3, "hint">;

def InlineKind : I32EnumAttr<"InlineKind", "inlineKind", [
  NoInline, AlwaysInline, InlineHint
]> {
  let cppNamespace = "::mlir::cir";
}

def InlineAttr : CIR_Attr<"Inline", "inline"> {
  let summary = "Inline attribute";
  let description = [{
    Inline attributes represents user directives.
  }];

  let parameters = (ins "InlineKind":$value);

  let assemblyFormat = [{
    `<` $value `>`
  }];

  let extraClassDeclaration = [{
    bool isNoInline() const { return getValue() == InlineKind::NoInline; };
    bool isAlwaysInline() const { return getValue() == InlineKind::AlwaysInline; };
    bool isInlineHint() const { return getValue() == InlineKind::InlineHint; };
  }];
}

#endif // MLIR_CIR_DIALECT_CIR_ATTRS
