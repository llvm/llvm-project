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

include "clang/CIR/Interfaces/ASTAttrInterfaces.td"

//===----------------------------------------------------------------------===//
// CIR Attrs
//===----------------------------------------------------------------------===//

class CIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class CIRUnitAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : CIR_Attr<name, attrMnemonic, traits> {
  let returnType = "bool";
  let defaultValue = "false";
  let valueType = NoneType;
  let isOptional = 1;
}

//===----------------------------------------------------------------------===//
// LangAttr
//===----------------------------------------------------------------------===//

def C : I32EnumAttrCase<"C", 1, "c">;
def CXX : I32EnumAttrCase<"CXX", 2, "cxx">;
def OpenCLC : I32EnumAttrCase<"OpenCLC", 3, "opencl_c">;

def SourceLanguage : I32EnumAttr<"SourceLanguage", "Source language", [
  C, CXX, OpenCLC
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
                        "Attribute":$elts,
                        "int":$trailingZerosNum);

  // Define a custom builder for the type; that removes the need to pass
  // in an MLIRContext instance, as it can be infered from the `type`.
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::cir::ArrayType":$type,
                                        "Attribute":$elts), [{
      int zeros = 0;
      auto typeSize = mlir::cast<mlir::cir::ArrayType>(type).getSize();
      if (auto str = mlir::dyn_cast<mlir::StringAttr>(elts))
        zeros = typeSize - str.size();
      else
        zeros = typeSize - mlir::cast<mlir::ArrayAttr>(elts).size();

      return $_get(type.getContext(), type, elts, zeros);
    }]>
  ];

  // Printing and parsing available in CIRDialect.cpp
  let hasCustomAssemblyFormat = 1;

  // Enable verifier.
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    bool hasTrailingZeros() const { return getTrailingZerosNum() != 0; };
  }];
}

//===----------------------------------------------------------------------===//
// ConstVectorAttr
//===----------------------------------------------------------------------===//

def ConstVectorAttr : CIR_Attr<"ConstVector", "const_vector",
                               [TypedAttrInterface]> {
  let summary = "A constant vector from ArrayAttr";
  let description = [{
    A CIR vector attribute is an array of literals of the specified attribute
    types.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
		        "ArrayAttr":$elts);

  // Define a custom builder for the type; that removes the need to pass in an
  // MLIRContext instance, as it can be inferred from the `type`.
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::cir::VectorType":$type,
		                        "ArrayAttr":$elts), [{
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
      IntType intType = mlir::cast<IntType>(type);
      mlir::APInt apValue(intType.getWidth(), value, intType.isSigned());
      return $_get(intType.getContext(), intType, apValue);
    }]>,
  ];
  let extraClassDeclaration = [{
    int64_t getSInt() const { return getValue().getSExtValue(); }
    uint64_t getUInt() const { return getValue().getZExtValue(); }
    bool isNullValue() const { return getValue() == 0; }
    uint64_t getBitWidth() const { return mlir::cast<IntType>(getType()).getWidth(); }
  }];
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// FPAttr
//===----------------------------------------------------------------------===//

def FPAttr : CIR_Attr<"FP", "fp", [TypedAttrInterface]> {
  let summary = "An attribute containing a floating-point value";
  let description = [{
    An fp attribute is a literal attribute that represents a floating-point
    value of the specified floating-point type.
  }];
  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "APFloat":$value);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const APFloat &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>,
  ];
  let extraClassDeclaration = [{
    static FPAttr getZero(mlir::Type type);
  }];
  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<` custom<FloatLiteral>($value, ref($type)) `>`
  }];
}

//===----------------------------------------------------------------------===//
// ComplexAttr
//===----------------------------------------------------------------------===//

def ComplexAttr : CIR_Attr<"Complex", "complex", [TypedAttrInterface]> {
  let summary = "An attribute that contains a constant complex value";
  let description = [{
    The `#cir.complex` attribute contains a constant value of complex number
    type. The `real` parameter gives the real part of the complex number and the
    `imag` parameter gives the imaginary part of the complex number.

    The `real` and `imag` parameter must be either an IntAttr or an FPAttr that
    contains values of the same CIR type.
  }];

  let parameters = (ins 
    AttributeSelfTypeParameter<"", "mlir::cir::ComplexType">:$type,
    "mlir::TypedAttr":$real, "mlir::TypedAttr":$imag);

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::cir::ComplexType":$type,
                                        "mlir::TypedAttr":$real,
                                        "mlir::TypedAttr":$imag), [{
      return $_get(type.getContext(), type, real, imag);
    }]>,
  ];

  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<` qualified($real) `,` qualified($imag) `>`
  }];
}

//===----------------------------------------------------------------------===//
// ConstPointerAttr
//===----------------------------------------------------------------------===//

def ConstPtrAttr : CIR_Attr<"ConstPtr", "ptr", [TypedAttrInterface]> {
  let summary = "Holds a constant pointer value";
  let parameters = (ins
    AttributeSelfTypeParameter<"", "::mlir::cir::PointerType">:$type,
    "mlir::IntegerAttr":$value);
  let description = [{
    A pointer attribute is a literal attribute that represents an integral
    value of a pointer type.
  }];
  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type, "mlir::IntegerAttr":$value), [{
      return $_get(type.getContext(), mlir::cast<mlir::cir::PointerType>(type), value);
    }]>,
    AttrBuilder<(ins "Type":$type,
                     "mlir::IntegerAttr":$value), [{
      return $_get($_ctxt, mlir::cast<mlir::cir::PointerType>(type), value);
    }]>,
  ];
  let extraClassDeclaration = [{
    bool isNullValue() const { return getValue().getInt() == 0; }
  }];

  let assemblyFormat = [{
    `<` custom<ConstPtr>($value) `>`
  }];
}

//===----------------------------------------------------------------------===//
// CmpThreeWayInfoAttr
//===----------------------------------------------------------------------===//

def CmpOrdering_Strong : I32EnumAttrCase<"Strong", 1, "strong">;
def CmpOrdering_Partial : I32EnumAttrCase<"Partial", 2, "partial">;

def CmpOrdering : I32EnumAttr<
  "CmpOrdering", "three-way comparison ordering kind",
  [CmpOrdering_Strong, CmpOrdering_Partial]
> {
  let cppNamespace = "::mlir::cir";
}

def CmpThreeWayInfoAttr : CIR_Attr<"CmpThreeWayInfo", "cmp3way_info"> {
  let summary = "Holds information about a three-way comparison operation";
  let description = [{
    The `#cmp3way_info` attribute contains information about a three-way
    comparison operation `cir.cmp3way`.

    The `ordering` parameter gives the ordering kind of the three-way comparison
    operation. It may be either strong ordering or partial ordering.

    Given the two input operands of the three-way comparison operation `lhs` and
    `rhs`, the `lt`, `eq`, `gt`, and `unordered` parameters gives the result
    value that should be produced by the three-way comparison operation when the
    ordering between `lhs` and `rhs` is `lhs < rhs`, `lhs == rhs`, `lhs > rhs`,
    or neither, respectively.
  }];

  let parameters = (ins "CmpOrdering":$ordering, "int64_t":$lt, "int64_t":$eq,
                        "int64_t":$gt,
                        OptionalParameter<"std::optional<int64_t>">:$unordered);

  let builders = [
    AttrBuilder<(ins "int64_t":$lt, "int64_t":$eq, "int64_t":$gt), [{
      return $_get($_ctxt, CmpOrdering::Strong, lt, eq, gt, std::nullopt);
    }]>,
    AttrBuilder<(ins "int64_t":$lt, "int64_t":$eq, "int64_t":$gt,
                     "int64_t":$unordered), [{
      return $_get($_ctxt, CmpOrdering::Partial, lt, eq, gt, unordered);
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Get attribute alias name for this attribute.
    std::string getAlias() const;
  }];

  let assemblyFormat = [{
    `<`
      $ordering `,`
      `lt` `=` $lt `,`
      `eq` `=` $eq `,`
      `gt` `=` $gt
      (`,` `unordered` `=` $unordered^)?
    `>`
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// DataMemberAttr
//===----------------------------------------------------------------------===//

def DataMemberAttr : CIR_Attr<"DataMember", "data_member",
                              [TypedAttrInterface]> {
  let summary = "Holds a constant data member pointer value";
  let parameters = (ins AttributeSelfTypeParameter<
                            "", "mlir::cir::DataMemberType">:$type,
                        OptionalParameter<
                            "std::optional<unsigned>">:$member_index);
  let description = [{
    A data member attribute is a literal attribute that represents a constant
    pointer-to-data-member value.

    The `member_index` parameter represents the index of the pointed-to member
    within its containing struct. It is an optional parameter; lack of this
    parameter indicates a null pointer-to-data-member value.

    Example:
    ```
    #ptr = #cir.data_member<1> : !cir.data_member<!s32i in !ty_22Point22>

    #null = #cir.data_member<null> : !cir.data_member<!s32i in !ty_22Point22>
    ```
  }];

  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<` ($member_index^):(`null`)? `>`
  }];

  let extraClassDeclaration = [{
    bool isNullPtr() const {
      return !getMemberIndex().has_value();
    }
  }];
}

//===----------------------------------------------------------------------===//
// MethodAttr
//===----------------------------------------------------------------------===//

def MethodAttr : CIR_Attr<"Method", "method", [TypedAttrInterface]> {
  let summary = "Holds a constant pointer-to-member-function value";
  let description = [{
    A method attribute is a literal attribute that represents a constant
    pointer-to-member-function value.

    If the member function is a non-virtual function, the `symbol` parameter
    gives the global symbol for the non-virtual member function.

    If the member function is a virtual function, the `vtable_offset` parameter
    gives the offset of the vtable entry corresponding to the virtual member
    function.

    `symbol` and `vtable_offset` cannot be present at the same time. If both of
    `symbol` and `vtable_offset` are not present, the attribute represents a
    null pointer constant.
  }];

  let parameters = (ins AttributeSelfTypeParameter<
                            "", "mlir::cir::MethodType">:$type,
                        OptionalParameter<
                            "std::optional<FlatSymbolRefAttr>">:$symbol,
                        OptionalParameter<
                            "std::optional<uint64_t>">:$vtable_offset);

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::cir::MethodType":$type), [{
      return $_get(type.getContext(), type, std::nullopt, std::nullopt);
    }]>,
    AttrBuilderWithInferredContext<(ins "mlir::cir::MethodType":$type,
                                        "FlatSymbolRefAttr":$symbol), [{
      return $_get(type.getContext(), type, symbol, std::nullopt);
    }]>,
    AttrBuilderWithInferredContext<(ins "mlir::cir::MethodType":$type,
                                        "uint64_t":$vtable_offset), [{
      return $_get(type.getContext(), type, std::nullopt, vtable_offset);
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;
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
    for the resulting struct, i.e, a GlobalViewAttr for symbol reference or
    an IntAttr for flags.

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
// StructLayoutAttr
//===----------------------------------------------------------------------===//

// Used to decouple layout information from the struct type. StructType's
// uses this attribute to cache that information.

def StructLayoutAttr : CIR_Attr<"StructLayout", "struct_layout"> {
  let summary = "ABI specific information about a struct layout";
  let description = [{
    Holds layout information often queried by !cir.struct users
    during lowering passes and optimizations.
  }];

  let parameters = (ins "unsigned":$size,
                        "unsigned":$alignment,
                        "bool":$padded,
                        "mlir::Type":$largest_member,
                        "mlir::ArrayAttr":$offsets);

  let builders = [
    AttrBuilderWithInferredContext<(ins "unsigned":$size,
                                        "unsigned":$alignment,
                                        "bool":$padded,
                                        "mlir::Type":$largest_member,
                                        "mlir::ArrayAttr":$offsets), [{
      return $_get(largest_member.getContext(), size, alignment, padded,
                   largest_member, offsets);
    }]>,
  ];

  let genVerifyDecl = 1;
  let assemblyFormat = [{
    `<`
      struct($size, $alignment, $padded, $largest_member, $offsets)
    `>`
  }];
}

//===----------------------------------------------------------------------===//
// DynamicCastInfoAttr
//===----------------------------------------------------------------------===//

def DynamicCastInfoAttr
    : CIR_Attr<"DynamicCastInfo", "dyn_cast_info"> {
  let summary = "ABI specific information about a dynamic cast";
  let description = [{
    Provide ABI specific information about a dynamic cast operation.

    The `srcRtti` and the `destRtti` parameters give the RTTI of the source
    struct type and the destination struct type, respectively.

    The `runtimeFunc` parameter gives the `__dynamic_cast` function which is
    provided by the runtime. The `badCastFunc` parameter gives the
    `__cxa_bad_cast` function which is also provided by the runtime.

    The `offsetHint` parameter gives the hint value that should be passed to the
    `__dynamic_cast` runtime function.
  }];

  let parameters = (ins GlobalViewAttr:$srcRtti,
                        GlobalViewAttr:$destRtti,
                        "FlatSymbolRefAttr":$runtimeFunc,
                        "FlatSymbolRefAttr":$badCastFunc,
                        IntAttr:$offsetHint);

  let builders = [
    AttrBuilderWithInferredContext<(ins "GlobalViewAttr":$srcRtti,
                                        "GlobalViewAttr":$destRtti,
                                        "FlatSymbolRefAttr":$runtimeFunc,
                                        "FlatSymbolRefAttr":$badCastFunc,
                                        "IntAttr":$offsetHint), [{
      return $_get(srcRtti.getContext(), srcRtti, destRtti, runtimeFunc,
                   badCastFunc, offsetHint);
    }]>,
  ];

  let genVerifyDecl = 1;
  let assemblyFormat = [{
    `<`
      qualified($srcRtti) `,` qualified($destRtti) `,`
      $runtimeFunc `,` $badCastFunc `,` qualified($offsetHint)
    `>`
  }];

  let extraClassDeclaration = [{
    /// Get attribute alias name for this attribute.
    std::string getAlias() const;
  }];
}

//===----------------------------------------------------------------------===//
// AddressSpaceAttr
//===----------------------------------------------------------------------===//

def AS_OffloadPrivate : I32EnumAttrCase<"offload_private", 1>;
def AS_OffloadLocal : I32EnumAttrCase<"offload_local", 2>;
def AS_OffloadGlobal : I32EnumAttrCase<"offload_global", 3>;
def AS_OffloadConstant : I32EnumAttrCase<"offload_constant", 4>;
def AS_OffloadGeneric : I32EnumAttrCase<"offload_generic", 5>;
def AS_Target : I32EnumAttrCase<"target", 6>;

def AddressSpaceAttr : CIR_Attr<"AddressSpace", "addrspace"> {

  let summary = "Address space attribute for pointer types";
  let description = [{
    The address space attribute is used in pointer types. It essentially
    provides a unified model on top of `clang::LangAS`, rather than LLVM address
    spaces.

    The representation is further simplified: `LangAS::Default` is encoded as
    a null attribute; many address spaces from different offloading languages
    are unified as `offload_*`; etc.

    The meaning of `value` parameter is defined as an extensible enum `Kind`,
    which encodes target AS as offset to the last language AS.
  }];

  let parameters = (ins "int32_t":$value);

  let assemblyFormat = [{
    `<` $value `>`
  }];

  let builders = [
    AttrBuilder<(ins "clang::LangAS":$langAS), [{
      assert(langAS != clang::LangAS::Default &&
        "Default address space is encoded as null attribute");
      return $_get($_ctxt, getValueFromLangAS(langAS).value());
    }]>
  ];

  let cppNamespace = "::mlir::cir";

  // The following codes implement these conversions:
  // clang::LangAS -> int32_t <-> text-form CIR

  // CIR_PointerType manipulates the parse- and stringify- methods to provide
  // simplified assembly format `custom<PointerAddrSpace>`.

  list<I32EnumAttrCase> langASCases = [
    AS_OffloadPrivate, AS_OffloadLocal, AS_OffloadGlobal, AS_OffloadConstant,
    AS_OffloadGeneric
  ];

  I32EnumAttrCase targetASCase = AS_Target;

  let extraClassDeclaration = [{
    static constexpr char kTargetKeyword[] = "}]#targetASCase.symbol#[{";
    static constexpr int32_t kFirstTargetASValue = }]#targetASCase.value#[{;
  
    bool isLang() const;
    bool isTarget() const;
    unsigned getTargetValue() const;

    /// Convert a clang LangAS to its corresponding CIR AS storage value. This
    /// helper does not perform any language-specific mappings (e.g. determining
    /// the default AS for offloading languages), so these must be handled in
    /// the caller.
    static std::optional<int32_t> getValueFromLangAS(clang::LangAS v);

    /// Helper methods for the assembly format `custom<PointerAddrSpace>`.
    static std::optional<int32_t> parseValueFromString(llvm::StringRef s);
    static std::optional<llvm::StringRef> stringifyValue(int32_t v);

    struct Kind {
    }]#!interleave(
      !foreach(case, langASCases,
        "static constexpr int32_t "#case.symbol#" = "#case.value#";"
      ), "\n"
    )#[{
    };
  }];

  let extraClassDefinition = [{
    bool $cppClass::isLang() const {
      return !isTarget();
    }

    bool $cppClass::isTarget() const {
      return getValue() >= kFirstTargetASValue;
    }

    unsigned $cppClass::getTargetValue() const {
      assert(isTarget() && "Not a target address space");
      return getValue() - kFirstTargetASValue;
    }

    std::optional<int32_t>
    $cppClass::parseValueFromString(llvm::StringRef str) {
      return llvm::StringSwitch<::std::optional<int32_t>>(str)
        }]
        #
        !interleave(
          !foreach(case, langASCases,
            ".Case(\""#case.symbol# "\", "#case.value # ")\n"
          ),
          "\n"
        )
        #
        [{
        // Target address spaces are not parsed here
        .Default(std::nullopt);
    }

    std::optional<llvm::StringRef>
    $cppClass::stringifyValue(int32_t value) {
      switch (value) {
      }]
      #
      !interleave(
        !foreach(case, langASCases,
          "case "#case.value
          # ": return \""#case.symbol # "\";" ),
        "\n"
      )
      #
      [{
      default:
        // Target address spaces are not processed here
        return std::nullopt;
      }
    }
  }];
}

//===----------------------------------------------------------------------===//
// AST Wrappers
//===----------------------------------------------------------------------===//

class AST<string name, string prefix, list<Trait> traits = []>
  : CIR_Attr<!strconcat("AST", name), !strconcat(prefix, ".ast"), traits> {
  string clang_name = !strconcat("const clang::", name, " *");

  let summary = !strconcat("Wraps a '", clang_name, "' AST node.");
  let description = [{
    Operations optionally refer to this node, they could be available depending
    on the CIR lowering stage. Whether it's attached to the appropriated
    CIR operation is delegated to the operation verifier.

    This always implies a non-null AST reference (verified).
  }];
  let parameters = (ins clang_name:$ast);

  // Printing and parsing available in CIRDialect.cpp
  let hasCustomAssemblyFormat = 1;

  // Enable verifier.
  let genVerifyDecl = 1;

  let extraClassDefinition = [{
    ::mlir::Attribute $cppClass::parse(::mlir::AsmParser &parser,
                                       ::mlir::Type type) {
      // We cannot really parse anything AST related at this point
      // since we have no serialization/JSON story.
      return $cppClass::get(parser.getContext(), nullptr);
    }

    void $cppClass::print(::mlir::AsmPrinter &printer) const {
      // Nothing to print besides the mnemonics.
    }

    LogicalResult $cppClass::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
        }] # clang_name # [{ decl) {
      return success();
    }
  }];
}

def ASTDeclAttr : AST<"Decl", "decl", [ASTDeclInterface]>;

def ASTFunctionDeclAttr : AST<"FunctionDecl", "function.decl",
  [ASTFunctionDeclInterface]>;

def ASTCXXMethodDeclAttr : AST<"CXXMethodDecl", "cxxmethod.decl",
  [ASTCXXMethodDeclInterface]>;

def ASTCXXConstructorDeclAttr : AST<"CXXConstructorDecl",
  "cxxconstructor.decl", [ASTCXXConstructorDeclInterface]>;

def ASTCXXConversionDeclAttr : AST<"CXXConversionDecl",
  "cxxconversion.decl", [ASTCXXConversionDeclInterface]>;

def ASTCXXDestructorDeclAttr : AST<"CXXDestructorDecl",
  "cxxdestructor.decl", [ASTCXXDestructorDeclInterface]>;

def ASTVarDeclAttr : AST<"VarDecl", "var.decl",
  [ASTVarDeclInterface]>;

def ASTTypeDeclAttr: AST<"TypeDecl", "type.decl",
  [ASTTypeDeclInterface]>;

def ASTTagDeclAttr : AST<"TagDecl", "tag.decl",
  [ASTTagDeclInterface]>;

def ASTRecordDeclAttr : AST<"RecordDecl", "record.decl",
  [ASTRecordDeclInterface]>;

def ASTExprAttr : AST<"Expr", "expr",
  [ASTExprInterface]>;

def ASTCallExprAttr : AST<"CallExpr", "call.expr",
  [ASTCallExprInterface]>;


//===----------------------------------------------------------------------===//
// VisibilityAttr
//===----------------------------------------------------------------------===//

def VK_Default : I32EnumAttrCase<"Default", 1, "default">; 
def VK_Hidden : I32EnumAttrCase<"Hidden", 2, "hidden">;
def VK_Protected : I32EnumAttrCase<"Protected", 3, "protected">;

def VisibilityKind : I32EnumAttr<"VisibilityKind", "C/C++ visibility", [
  VK_Default, VK_Hidden, VK_Protected
]> {
  let cppNamespace = "::mlir::cir";
}

def VisibilityAttr : CIR_Attr<"Visibility", "visibility"> {
  let summary = "Visibility attribute";
  let description = [{
    Visibility attributes.
  }];
  let parameters = (ins "VisibilityKind":$value);

  let assemblyFormat = [{
    $value
  }];

  let builders = [
    AttrBuilder<(ins CArg<"VisibilityKind", "cir::VisibilityKind::Default">:$value), [{
      return $_get($_ctxt, value);
    }]>
  ];

  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    bool isDefault() const { return getValue() == VisibilityKind::Default; };
    bool isHidden() const { return getValue() == VisibilityKind::Hidden; };
    bool isProtected() const { return getValue() == VisibilityKind::Protected; };
  }]; 
}


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

def OptNoneAttr : CIRUnitAttr<"OptNone", "optnone"> {
  let storageType = [{ OptNoneAttr }];
}

def NoThrowAttr : CIRUnitAttr<"NoThrow", "nothrow"> {
  let storageType = [{ NoThrowAttr }];
}

class CIR_GlobalCtorDtor<string name, string attrMnemonic,
                         string sum, string desc>
    : CIR_Attr<"Global" # name, "global_" # attrMnemonic> {
  let summary = sum;
  let description = desc;

  let parameters = (ins "StringAttr":$name, "int":$priority);
  let assemblyFormat = [{
    `<`
      $name `,` $priority
    `>`
  }];
  let builders = [
    AttrBuilder<(ins "StringRef":$name,
                      CArg<"int", "65536">:$priority), [{
      return $_get($_ctxt, StringAttr::get($_ctxt, name), priority);
    }]>
  ];
  let extraClassDeclaration = [{
    bool isDefaultPriority() const { return getPriority() == 65536; };
  }];
  let skipDefaultBuilders = 1;
}

def GlobalCtorAttr : CIR_GlobalCtorDtor<"Ctor", "ctor",
  "Marks a function as a global constructor",
  "A function with this attribute executes before main()"
>;
def GlobalDtorAttr : CIR_GlobalCtorDtor<"Dtor", "dtor",
  "Marks a function as a global destructor",
  "A function with this attribute excutes before module unloading"
>;

def BitfieldInfoAttr : CIR_Attr<"BitfieldInfo", "bitfield_info"> {
  let summary = "Represents a bit field info";
  let description = [{
    Holds the next information about bitfields: name, storage type, a bitfield size
    and position in the storage, if the bitfield is signed or not.
  }];
  let parameters = (ins "StringAttr":$name,
                        "Type":$storage_type,
                        "uint64_t":$size,
                        "uint64_t":$offset,
                        "bool":$is_signed);

  let assemblyFormat = "`<` struct($name, $storage_type, $size, $offset, $is_signed) `>`";

  let builders = [
    AttrBuilder<(ins "StringRef":$name,
                     "Type":$storage_type,
                     "uint64_t":$size,
                     "uint64_t":$offset,
                     "bool":$is_signed
                     ), [{
      return $_get($_ctxt, StringAttr::get($_ctxt, name), storage_type, size, offset, is_signed);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// AnnotationAttr
//===----------------------------------------------------------------------===//

def AnnotationAttr : CIR_Attr<"Annotation", "annotation"> {
  let summary = "Annotation attribute for global variables and functions";
  let description = [{
    Represent C/C++ attribute of annotate in CIR.
    Example C code:
    ```
     int *a __attribute__((annotate("testptr", "21", 12 )));
    ```
    In this example code, the `AnnotationAttr` has annotation name "testptr",
    and arguments "21" and 12 constitutes an `ArrayAttr` type parameter `args`
    for global variable `a`.
    In CIR, the attribute for above annotation looks like:
    ```
    [#cir.annotation<name = "withargs", args = ["21", 12 : i32]>]
    ```
  }];

  // The parameter args is empty when there is no arg.
  let parameters = (ins "StringAttr":$name,
                        "ArrayAttr":$args);

  let assemblyFormat = "`<` struct($name, $args) `>`";                  

  let extraClassDeclaration = [{
    bool isNoArgs() const { return getArgs().empty(); };
  }];
}

//===----------------------------------------------------------------------===//
// GlobalAnnotationValuesAttr
//===----------------------------------------------------------------------===//

def GlobalAnnotationValuesAttr : CIR_Attr<"GlobalAnnotationValues",
                                          "global_annotations"> {
  let summary = "Array of annotations, each element consists of name of"
                "a global var or func and one of its annotations";
  let description = [{
    This is annotation value array, which holds the annotation
    values for all global variables and functions in a module.
    This array is used to create the initial value of a global annotation
    metadata variable in LLVM IR.
    Example C code:
    ```
    double *a __attribute__((annotate("withargs", "21", 12 )));
    int *b __attribute__((annotate("withargs", "21", 12 )));
    void *c __attribute__((annotate("noargvar")));
    void foo(int i) __attribute__((annotate("noargfunc"))) {}
    ```
    After CIR lowering prepare pass, compiler generates a 
    `GlobalAnnotationValuesAttr` like the following:
    ```
    #cir<global_annotations [
      ["a", #cir.annotation<name = "withargs", args = ["21", 12 : i32]>],
      ["b", #cir.annotation<name = "withargs", args = ["21", 12 : i32]>],
      ["c", #cir.annotation<name = "noargvar", args = []>],
      ["foo", #cir.annotation<name = "noargfunc", args = []>]]>
    ```
  }];

  let parameters = (ins "ArrayAttr":$annotations);

  let assemblyFormat = [{ $annotations }];

  // Enable verifier.
  let genVerifyDecl = 1;
}

include "clang/CIR/Dialect/IR/CIROpenCLAttrs.td"

#endif // MLIR_CIR_DIALECT_CIR_ATTRS
