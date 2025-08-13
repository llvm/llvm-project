//===----------------------------------------------------------------------===//
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

#ifndef CLANG_CIR_DIALECT_IR_CIRATTRS_TD
#define CLANG_CIR_DIALECT_IR_CIRATTRS_TD

include "mlir/IR/BuiltinAttributeInterfaces.td"

include "clang/CIR/Dialect/IR/CIRAttrConstraints.td"
include "clang/CIR/Dialect/IR/CIRDialect.td"
include "clang/CIR/Dialect/IR/CIREnumAttr.td"

//===----------------------------------------------------------------------===//
// CIR Attrs
//===----------------------------------------------------------------------===//

class CIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class CIR_TypedAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : CIR_Attr<name, attrMnemonic, !listconcat(traits, [TypedAttrInterface])> {

  let parameters = (ins AttributeSelfTypeParameter<"">:$type);

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type), [{
      return $_get(type.getContext(), type);
    }]>
  ];

  let assemblyFormat = [{}];
}

class CIR_UnitAttr<string name, string attrMnemonic, list<Trait> traits = []>
    : CIR_Attr<name, attrMnemonic, traits> {
  let returnType = "bool";
  let defaultValue = "false";
  let valueType = NoneType;
  let isOptional = 1;
}

//===----------------------------------------------------------------------===//
// OptInfoAttr
//===----------------------------------------------------------------------===//

def CIR_OptInfoAttr : CIR_Attr<"OptInfo", "opt_info"> {
  let summary =
    "A module-level attribute that holds the optimization information";
  let description = [{
    The `#cir.opt_info` attribute holds optimization related information. For
    now this attribute is a module-level attribute that gets attached to the
    module operation during CIRGen.

    The `level` parameter gives the optimization level. It must be an integer
    between 0 and 3, inclusive. It corresponds to the `OptimizationLevel` field
    within the `clang::CodeGenOptions` structure.

    The `size` parameter gives the code size optimization level. It must be an
    integer between 0 and 2, inclusive. It corresponds to the `OptimizeSize`
    field within the `clang::CodeGenOptions` structure.

    The `level` and `size` parameters correspond to the optimization level
    command line options passed to clang driver. The table below lists the
    current correspondance relationship:

    | Flag             | `level` | `size` |
    |------------------|---------|--------|
    | `-O0` or nothing | 0       | 0      |
    | `-O1`            | 1       | 0      |
    | `-O2`            | 2       | 0      |
    | `-O3`            | 3       | 0      |
    | `-Os`            | 2       | 1      |
    | `-Oz`            | 2       | 2      |

    Examples:

    ```mlir
    #cir.opt_info<level = 2, size = 0>  // -O2
    ```
  }];

  let parameters = (ins "unsigned":$level, "unsigned":$size);

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
  let genVerifyDecl = 1;
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
                        "", "cir::BoolType">:$type,
                    "bool":$value);

  let builders = [
    AttrBuilder<(ins "bool":$value), [{
      return $_get($_ctxt, cir::BoolType::get($_ctxt), value);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $value `>`
  }];
}

//===----------------------------------------------------------------------===//
// ZeroAttr
//===----------------------------------------------------------------------===//

def CIR_ZeroAttr : CIR_TypedAttr<"Zero", "zero"> {
  let summary = "Attribute to represent zero initialization";
  let description = [{
    The ZeroAttr is used to indicate zero initialization on structs.
  }];
}

//===----------------------------------------------------------------------===//
// UndefAttr
//===----------------------------------------------------------------------===//

def CIR_UndefAttr : CIR_TypedAttr<"Undef", "undef"> {
  let summary = "Represent an undef constant";
  let description = [{
    The UndefAttr represents an undef constant, corresponding to LLVM's notion
    of undef.
  }];
}

//===----------------------------------------------------------------------===//
// PoisonAttr
//===----------------------------------------------------------------------===//

def CIR_PoisonAttr : CIR_TypedAttr<"Poison", "poison"> {
  let summary = "Represent a typed poison constant";
  let description = [{
    The PoisonAttr represents a typed poison constant, corresponding to LLVM's
    notion of poison.
  }];
}

//===----------------------------------------------------------------------===//
// IntegerAttr
//===----------------------------------------------------------------------===//

def CIR_IntAttr : CIR_Attr<"Int", "int", [TypedAttrInterface]> {
  let summary = "An attribute containing an integer value";
  let description = [{
    An integer attribute is a literal attribute that represents an integral
    value of the specified integer type.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "cir::IntTypeInterface">:$type,
    APIntParameter<"">:$value
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type,
                                        "const llvm::APInt &":$value), [{
      auto intType = mlir::cast<cir::IntTypeInterface>(type);
      return $_get(type.getContext(), intType, value);
    }]>,
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type,
                                        "int64_t":$value), [{
      auto intType = mlir::cast<cir::IntTypeInterface>(type);
      mlir::APInt apValue(intType.getWidth(), value, intType.isSigned());
      return $_get(intType.getContext(), intType, apValue);
    }]>,
  ];

  let extraClassDeclaration = [{
    int64_t getSInt() const;
    uint64_t getUInt() const;
    bool isNullValue() const;
    bool isSigned() const;
    bool isUnsigned() const;
    uint64_t getBitWidth() const;
  }];

 let extraClassDefinition = [{
    int64_t $cppClass::getSInt() const {
      return getValue().getSExtValue();
    }
    uint64_t $cppClass::getUInt() const {
      return getValue().getZExtValue();
    }
    bool $cppClass::isNullValue() const {
      return getValue() == 0;
    }
    bool $cppClass::isSigned() const {
      return mlir::cast<IntTypeInterface>(getType()).isSigned();
    }
    bool $cppClass::isUnsigned() const {
      return mlir::cast<IntTypeInterface>(getType()).isUnsigned();
    }
    uint64_t $cppClass::getBitWidth() const {
      return mlir::cast<IntTypeInterface>(getType()).getWidth();
    }
  }];

  let assemblyFormat = [{
    `<` custom<IntLiteral>($value, ref($type)) `>`
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// FPAttr
//===----------------------------------------------------------------------===//

def CIR_FPAttr : CIR_Attr<"FP", "fp", [TypedAttrInterface]> {
  let summary = "An attribute containing a floating-point value";
  let description = [{
    An fp attribute is a literal attribute that represents a floating-point
    value of the specified floating-point type. Supporting only CIR FP types.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "::cir::FPTypeInterface">:$type,
    APFloatParameter<"">:$value
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type,
                                        "const llvm::APFloat &":$value), [{
      return $_get(type.getContext(), mlir::cast<FPTypeInterface>(type), value);
    }]>
  ];

  let extraClassDeclaration = [{
    static FPAttr getZero(mlir::Type type);
  }];

  let assemblyFormat = [{
    `<` custom<FloatLiteral>($value, ref($type)) `>`
  }];

  let genVerifyDecl = 1;
}


//===----------------------------------------------------------------------===//
// ConstArrayAttr
//===----------------------------------------------------------------------===//

def CIR_ConstArrayAttr : CIR_Attr<"ConstArray", "const_array", [
  TypedAttrInterface
]> {
  let summary = "A constant array from ArrayAttr or StringRefAttr";
  let description = [{
    An CIR array attribute is an array of literals of the specified attr types.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "mlir::Attribute":$elts,
                        "int":$trailingZerosNum);

  // Define a custom builder for the type; that removes the need to pass
  // in an MLIRContext instance, as it can be infered from the `type`.
  let builders = [
    AttrBuilderWithInferredContext<(ins "cir::ArrayType":$type,
                                        "mlir::Attribute":$elts), [{
      int zeros = 0;
      auto typeSize = mlir::cast<cir::ArrayType>(type).getSize();
      if (auto str = mlir::dyn_cast<mlir::StringAttr>(elts))
        zeros = typeSize - str.size();
      else
        zeros = typeSize - mlir::cast<mlir::ArrayAttr>(elts).size();

      return $_get(type.getContext(), type, elts, zeros);
    }]>,
    AttrBuilderWithInferredContext<(ins "cir::ArrayType":$type,
                                        "mlir::Attribute":$elts,
                                        "int":$trailingZerosNum), [{
      return $_get(type.getContext(), type, elts, trailingZerosNum);
    }]>
  ];

  // Printing and parsing available in CIRAttrs.cpp
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

def CIR_ConstVectorAttr : CIR_Attr<"ConstVector", "const_vector", [
  TypedAttrInterface
]> {
  let summary = "A constant vector from ArrayAttr";
  let description = [{
    A CIR vector attribute is an array of literals of the specified attribute
    types.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
		        "mlir::ArrayAttr":$elts);

  // Define a custom builder for the type; that removes the need to pass in an
  // MLIRContext instance, as it can be inferred from the `type`.
  let builders = [
    AttrBuilderWithInferredContext<(ins "cir::VectorType":$type,
		                        "mlir::ArrayAttr":$elts), [{
      return $_get(type.getContext(), type, elts);
    }]>
  ];

  let assemblyFormat = [{
    `<` $elts `>`
  }];

  // Enable verifier.
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// ConstPtrAttr
//===----------------------------------------------------------------------===//

def CIR_ConstPtrAttr : CIR_Attr<"ConstPtr", "ptr", [TypedAttrInterface]> {
  let summary = "Holds a constant pointer value";
  let parameters = (ins
    AttributeSelfTypeParameter<"", "::cir::PointerType">:$type,
    "mlir::IntegerAttr":$value);
  let description = [{
    A pointer attribute is a literal attribute that represents an integral
    value of a pointer type.
  }];
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type,
                                        "mlir::IntegerAttr":$value), [{
      return $_get(type.getContext(), mlir::cast<cir::PointerType>(type),
                   value);
    }]>
  ];
  let extraClassDeclaration = [{
    bool isNullValue() const { return getValue().getInt() == 0; }
  }];

  let assemblyFormat = [{
    `<` custom<ConstPtr>($value) `>`
  }];
}

//===----------------------------------------------------------------------===//
// GlobalViewAttr
//===----------------------------------------------------------------------===//

def CIR_GlobalViewAttr : CIR_Attr<"GlobalView", "global_view", [
  TypedAttrInterface
]> {
  let summary = "Provides constant access to a global address";
  let description = [{
    Get constant address of global `symbol`. It provides a way to access globals
    from other global and always produces a pointer.

    The type of the input symbol can be different from `#cir.global_view`
    output type, since a given view of the global might require a static
    cast for initializing other globals.

    The result type of this attribute may be an integer type. In such a case,
    the pointer to the referenced global is casted to an integer and this
    attribute represents the casted result.

    Example:

    ```
      cir.global external @s = @".str2": !cir.ptr<i8>
      cir.global external @x = #cir.global_view<@s> : !cir.ptr<i8>
      cir.global external @s_addr = #cir.global_view<@s> : !s64i
    ```
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "mlir::FlatSymbolRefAttr":$symbol);

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::Type":$type,
                                        "mlir::FlatSymbolRefAttr":$symbol), [{
      return $_get(type.getContext(), type, symbol);
    }]>
  ];

  // let genVerifyDecl = 1;
  let assemblyFormat = [{
    `<`
      $symbol
    `>`
  }];
}

//===----------------------------------------------------------------------===//
// ConstComplexAttr
//===----------------------------------------------------------------------===//

def CIR_ConstComplexAttr : CIR_Attr<"ConstComplex", "const_complex", [
  TypedAttrInterface
]> {
  let summary = "An attribute that contains a constant complex value";
  let description = [{
    The `#cir.const_complex` attribute contains a constant value of complex
    number type. The `real` parameter gives the real part of the complex number
    and the `imag` parameter gives the imaginary part of the complex number.

    The `real` and `imag` parameters must both reference the same type and must
    be either IntAttr or FPAttr.

    ```mlir
    %ci = #cir.const_complex<#cir.int<1> : !s32i, #cir.int<2> : !s32i>
        : !cir.complex<!s32i>
    %cf = #cir.const_complex<#cir.fp<1.000000e+00> : !cir.float,
        #cir.fp<2.000000e+00> : !cir.float> : !cir.complex<!cir.float>
    ```
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "cir::ComplexType">:$type,
    CIR_AnyIntOrFloatAttr:$real,
    CIR_AnyIntOrFloatAttr:$imag
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::TypedAttr":$real,
                                        "mlir::TypedAttr":$imag), [{
      auto type = cir::ComplexType::get(real.getType());
      return $_get(type.getContext(), type, real, imag);
    }]>,
  ];

  let genVerifyDecl = 1;

  let assemblyFormat = [{
    `<` qualified($real) `,` qualified($imag) `>`
  }];
}

//===----------------------------------------------------------------------===//
// VisibilityAttr
//===----------------------------------------------------------------------===//

def CIR_VisibilityKind : CIR_I32EnumAttr<"VisibilityKind", "C/C++ visibility", [
  I32EnumAttrCase<"Default", 0, "default">,
  I32EnumAttrCase<"Hidden", 1, "hidden">,
  I32EnumAttrCase<"Protected", 2, "protected">
]> {
  let genSpecializedAttr = 0;
}

def CIR_VisibilityAttr : CIR_EnumAttr<CIR_VisibilityKind, "visibility"> {
  let summary = "Visibility attribute";
  let description = [{
    Visibility attributes.
  }];

  let cppClassName = "VisibilityAttr";

  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins CArg<"VisibilityKind",
                          "cir::VisibilityKind::Default">:$value), [{
      return $_get($_ctxt, value);
    }]>
  ];

  let assemblyFormat = [{
    $value
  }];

  let extraClassDeclaration = [{
    bool isDefault() const { return getValue() == VisibilityKind::Default; };
    bool isHidden() const { return getValue() == VisibilityKind::Hidden; };
    bool isProtected() const { return getValue() == VisibilityKind::Protected; };
  }];
}

//===----------------------------------------------------------------------===//
// BitfieldInfoAttr
//===----------------------------------------------------------------------===//

def CIR_BitfieldInfoAttr : CIR_Attr<"BitfieldInfo", "bitfield_info"> {
  let summary = "Represents info for a bit-field member";
  let description = [{
    Holds the following information about bitfields: name, storage type, size
    and position in the storage, and signedness.
    Example:
      Given the following struct with bitfields:
        ```c++
        typedef struct {
          int a : 4;
          int b : 27;
          int c : 17;
          int d : 2;
          int e : 15;
        } S;
        ```

      The CIR representation of the struct `S` might look like:
      ```mlir
        !rec_S = !cir.record<struct "S" packed padded {!u64i, !u16i,
                                                       !cir.array<!u8i x 2>}>
      ```
      And the bitfield info attribute for member `a` would be:
      ```mlir
        #bfi_a = #cir.bitfield_info<name = "a", storage_type = !u64i,
                                    size = 4, offset = 0, is_signed = true>
      ```

      This metadata describes that field `a` is stored in a 64-bit integer,
      is 4 bits wide, starts at offset 0, and is signed.
  }];
  let parameters = (ins "mlir::StringAttr":$name,
                        "mlir::Type":$storage_type,
                        "uint64_t":$size,
                        "uint64_t":$offset,
                        "bool":$is_signed);

  let assemblyFormat = [{`<` struct($name,
                                    $storage_type,
                                    $size,
                                    $offset,
                                    $is_signed)
                         `>`
                        }];

  let builders = [
    AttrBuilder<(ins "llvm::StringRef":$name,
                     "mlir::Type":$storage_type,
                     "uint64_t":$size,
                     "uint64_t":$offset,
                     "bool":$is_signed
                     ), [{
      return $_get($_ctxt, mlir::StringAttr::get($_ctxt, name), storage_type,
                   size, offset, is_signed);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// AddressPointAttr
//===----------------------------------------------------------------------===//

def CIR_AddressPointAttr : CIR_Attr<"AddressPoint", "address_point"> {
  let summary = "Address point attribute";

  let description = [{
    Attribute specifying the address point within a C++ virtual table (vtable).

    The `index` (vtable index) parameter identifies which vtable to use within a
    vtable group, while the `offset` (address point index) specifies the offset
    within that vtable where the address begins.

    Example:
    ```mlir
    cir.global linkonce_odr @_ZTV1B = ...
    ...
    %3 = cir.vtable.address_point(@_ZTV1B,
                                  address_point = <index = 0, offset = 2>)
                                 : !cir.vptr
    ```
  }];

  let parameters = (ins "int32_t":$index,
                        "int32_t":$offset);

  let assemblyFormat = [{
    `<` struct($index, $offset) `>`
  }];
}

#endif // CLANG_CIR_DIALECT_IR_CIRATTRS_TD
