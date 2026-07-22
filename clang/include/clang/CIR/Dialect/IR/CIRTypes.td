//===- CIRTypes.td - CIR dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRTYPES_TD
#define CLANG_CIR_DIALECT_IR_CIRTYPES_TD

include "clang/CIR/Dialect/IR/CIRDialect.td"
include "clang/CIR/Dialect/IR/CIREnumAttr.td"
include "clang/CIR/Dialect/IR/CIRTypeConstraints.td"
include "clang/CIR/Interfaces/CIRTypeInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// CIR Types
//===----------------------------------------------------------------------===//

class CIR_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<CIR_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// IntType
//===----------------------------------------------------------------------===//

def CIR_IntType : CIR_Type<"Int", "int", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
    DeclareTypeInterfaceMethods<CIR_SizedTypeInterface>,
    DeclareTypeInterfaceMethods<CIR_IntTypeInterface>,
]> {
  let summary = "Integer type with arbitrary precision up to a fixed limit";
  let description = [{
    CIR type that represents integer types with arbitrary precision, including
    standard integral types such as `int` and `long`, extended integral types
    such as `__int128`, and arbitrary width types such as `_BitInt(n)`.

    Those integer types that are directly available in C/C++ standard are called
    fundamental integer types. Said types are: `signed char`, `short`, `int`,
    `long`, `long long`, and their unsigned variations.

    Examples: `!cir.int<s, 32>`, `!cir.int<u, 64>`, `!cir.int<s, 128, bitint>`
  }];
  let parameters = (ins "unsigned":$width, "bool":$is_signed,
                        DefaultValuedParameter<"bool", "false">:$is_bit_int);
  let builders = [
    TypeBuilder<(ins "unsigned":$width, "bool":$is_signed), [{
      return $_get($_ctxt, width, is_signed, /*is_bit_int=*/false);
    }]>,
  ];
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Return true if this is a signed integer type.
    bool isSigned() const { return getIsSigned(); }
    /// Return true if this is an unsigned integer type.
    bool isUnsigned() const { return !getIsSigned(); }
    /// Return true if this is a _BitInt type.
    bool isBitInt() const { return getIsBitInt(); }
    /// Return type alias.
    std::string getAlias() const {
      std::string alias =
          (isSigned() ? 's' : 'u') + std::to_string(getWidth()) + 'i';
      if (isBitInt())
        alias += "_bitint";
      return alias;
    }
    /// Return true if this is a fundamental integer type (i.e. signed or
    /// unsigned integer types whose bit width is 8, 16, 32, or 64).
    /// _BitInt types are never fundamental even if their width matches.
    bool isFundamental() const {
      return isFundamentalIntType(*this);
    }
    bool isSignedFundamental() const {
      return isFundamentalSIntType(*this);
    }
    bool isUnsignedFundamental() const {
      return isFundamentalUIntType(*this);
    }

    /// Returns a minimum bitwidth of cir::IntType
    static unsigned minBitwidth() { return 1; }
    /// Returns a maximum bitwidth of cir::IntType.
    /// Matches llvm::IntegerType::MAX_INT_BITS (1 << 23).
    static unsigned maxBitwidth() { return (1 << 23); }
  }];
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// FloatType
//===----------------------------------------------------------------------===//

class CIR_FloatType<string name, string mnemonic> : CIR_Type<name, mnemonic, [
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
  DeclareTypeInterfaceMethods<CIR_FPTypeInterface>,
  DeclareTypeInterfaceMethods<CIR_SizedTypeInterface>
]>;

def CIR_Single : CIR_FloatType<"Single", "float"> {
  let summary = "CIR single-precision 32-bit float type";
  let description = [{
    A 32-bit floating-point type whose format is IEEE-754 `binary32`.  It
    represents the types `float`, `_Float32`, and `std::float32_t` in C and C++.
  }];
}

def CIR_Double : CIR_FloatType<"Double", "double"> {
  let summary = "CIR double-precision 64-bit float type";
  let description = [{
    A 64-bit floating-point type whose format is IEEE-754 `binary64`. It
    represents the types `double', '_Float64`, `std::float64_t`, and `_Float32x`
    in C and C++.  This is the underlying type for `long double` on some
    platforms, including Windows.
  }];
}

def CIR_FP16 : CIR_FloatType<"FP16", "f16"> {
  let summary = "CIR half-precision 16-bit float type";
  let description = [{
    A 16-bit floating-point type whose format is IEEE-754 `binary16`. It
    represents the types '_Float16` and `std::float16_t` in C and C++.
  }];
}

def CIR_BFloat16 : CIR_FloatType<"BF16", "bf16"> {
  let summary = "CIR bfloat16 16-bit float type";
  let description = [{
    A 16-bit floating-point type in the bfloat16 format, which is the same as
    IEEE `binary32` except that the lower 16 bits of the mantissa are missing.
    It represents the type `std::bfloat16_t` in C++, also spelled `__bf16` in
    some implementations.
  }];
}

def CIR_FP80 : CIR_FloatType<"FP80", "f80"> {
  let summary = "CIR x87 80-bit float type";
  let description = [{
    An 80-bit floating-point type in the x87 extended precision format.  The
    size and alignment of the type are both 128 bits, even though only 80 of
    those bits are used.  This is the underlying type for `long double` on Linux
    x86 platforms, and it is available as an extension in some implementations.
  }];
}

def CIR_FP128 : CIR_FloatType<"FP128", "f128"> {
  let summary = "CIR quad-precision 128-bit float type";
  let description = [{
    A 128-bit floating-point type whose format is IEEE-754 `binary128`. It
    represents the types `_Float128` and `std::float128_t` in C and C++, and the
    extension `__float128` in some implementations.  This is the underlying type
    for `long double` on some platforms including Linux Arm.
  }];
}

def CIR_LongDouble : CIR_FloatType<"LongDouble", "long_double"> {
  let summary = "CIR float type for `long double`";
  let description = [{
    A floating-point type that represents the `long double` type in C and C++.

    The underlying floating-point format of a `long double` value depends on the
    target platform and the implementation. The `underlying` parameter specifies
    the CIR floating-point type that corresponds to this format. Underlying
    types of IEEE 64-bit, IEEE 128-bit, x87 80-bit, and IBM's double-double
    format are all in use.
  }];

  let parameters = (ins AnyTypeOf<[CIR_Double, CIR_FP80, CIR_FP128],
    "expects !cir.double, !cir.fp80 or !cir.fp128">:$underlying);

  let assemblyFormat = [{
    `<` $underlying `>`
  }];
}

//===----------------------------------------------------------------------===//
// ComplexType
//===----------------------------------------------------------------------===//

def CIR_ComplexType : CIR_Type<"Complex", "complex", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
    DeclareTypeInterfaceMethods<CIR_SizedTypeInterface>
]> {
  let summary = "CIR complex type";
  let description = [{
    CIR type that represents a C complex number. `cir.complex` models the C type
    `T _Complex`.

    `cir.complex` type is not directly mapped to `std::complex`.

    The type models complex values, per C99 6.2.5p11. It supports the C99
    complex float types as well as the GCC integer complex extensions.

    The parameter `elementType` gives the type of the real and imaginary part of
    the complex number. `elementType` must be either a CIR integer type or a CIR
    floating-point type.

    ```
    !cir.complex<!s32i>
    !cir.complex<!cir.float>
    ```
  }];

  let parameters = (ins CIR_AnyIntOrFloatType:$elementType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $elementType `>`
  }];

  let extraClassDeclaration = [{
    bool isFloatingPointComplex() const {
      return isAnyFloatingPointType(getElementType());
    }

    bool isIntegerComplex() const {
      return mlir::isa<cir::IntType>(getElementType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

def CIR_PointerType : CIR_Type<"Pointer", "ptr", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
    DeclareTypeInterfaceMethods<CIR_SizedTypeInterface>
]> {
  let summary = "CIR pointer type";
  let description = [{
    The `!cir.ptr` type is a typed pointer type. It is used to represent
    pointers to objects in C/C++. The type of the pointed-to object is given by
    the `pointee` parameter. The `addrSpace` parameter is an optional address
    space attribute that specifies the address space of the pointer. If not
    specified, the pointer is assumed to be in the default address space.

    The `!cir.ptr` type can point to any type, including fundamental types,
    records, arrays, vectors, functions, and other pointers. It can also point
    to incomplete types, such as incomplete records.

    Examples:

    ```
    !cir.ptr<!cir.int<u, 8>>
    !cir.ptr<!cir.float>
    !cir.ptr<!cir.record<struct "MyStruct">>
    !cir.ptr<!cir.int<u, 8>, target_address_space(1)>
    !cir.ptr<!cir.record<struct "MyStruct">, target_address_space(5)>
    ```
  }];

  let parameters = (ins
    "mlir::Type":$pointee,
    OptionalParameter<
        "mlir::ptr::MemorySpaceAttrInterface">:$addrSpace
  );

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "mlir::Type":$pointee,
       CArg<"mlir::ptr::MemorySpaceAttrInterface", "{}">:$addrSpace), [{
        // Drop default address space and replace with empty attribute.
        addrSpace = cir::normalizeDefaultAddressSpace(addrSpace);
        return $_get(pointee.getContext(), pointee, addrSpace);
    }]>,
    TypeBuilder<(ins
      "mlir::Type":$pointee,
      CArg<"mlir::ptr::MemorySpaceAttrInterface", "{}">:$addrSpace), [{
        // Drop default address space and replace with empty attribute.
        addrSpace = cir::normalizeDefaultAddressSpace(addrSpace);
        return $_get($_ctxt, pointee, addrSpace);
    }]>
  ];

  let assemblyFormat = [{
    `<`
      $pointee
      ( `,` ` ` custom<AddressSpaceValue>($addrSpace)^ )?
    `>`
  }];

  let extraClassDeclaration = [{
    template <typename ...Types>
    bool isPtrTo() const {
      return mlir::isa< Types... >(getPointee());
    }

    bool isVoidPtr() const {
      return isPtrTo<cir::VoidType>();
    }

    template <typename ...Types>
    bool isPtrToPtrTo() const {
      if (auto ptrType = mlir::dyn_cast<cir::PointerType>(getPointee()))
        return ptrType.isPtrTo<Types...>();
      return false;
    }

    bool isPtrTo(mlir::Type type) const {
      return getPointee() == type;
    }

    bool isPtrToPtrTo(mlir::Type type) const {
      if (auto ptrType = mlir::dyn_cast<cir::PointerType>(getPointee()))
        return ptrType.isPtrTo(type);
      return false;
    }
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// CIR_DataMemberType
//===----------------------------------------------------------------------===//

def CIR_DataMemberType : CIR_Type<"DataMember", "data_member", [
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
  DeclareTypeInterfaceMethods<CIR_SizedTypeInterface>
]> {
  let summary = "CIR type that represents a pointer-to-data-member in C++";
  let description = [{
    `cir.data_member` models a pointer-to-data-member in C++. Values of this
    type are essentially offsets of the pointed-to member within one of its
    containing record.
  }];

  let parameters = (ins "mlir::Type":$member_ty,
                        "cir::RecordType":$class_ty);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "mlir::Type":$member_ty, "cir::RecordType":$class_ty
    ), [{
      return $_get(member_ty.getContext(), member_ty, class_ty);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $member_ty `in` $class_ty `>`
  }];
}

//===----------------------------------------------------------------------===//
// CIR_VPtrType
//===----------------------------------------------------------------------===//

def CIR_VPtrType : CIR_Type<"VPtr", "vptr", [
    DeclareTypeInterfaceMethods<CIR_SizedTypeInterface>,
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
]> {
  let summary = "CIR type that is used for the vptr member of C++ objects";
  let description = [{
    `cir.vptr` is a special type used as the type for the vptr member of a C++
    object. This avoids using arbitrary pointer types to declare vptr values
    and allows stronger type-based checking for operations that use or provide
    access to the vptr.

    This type will be the element type of the 'vptr' member of structures that
    require a vtable pointer. The `cir.vtable.address_point` operation returns
    this type. The `cir.vtable.get_vptr` operations returns a pointer to this
    type. This pointer may be passed to the `cir.vtable.get_virtual_fn_addr`
    operation to get the address of a virtual function pointer.

    The pointer may also be cast to other pointer types in order to perform
    pointer arithmetic based on information encoded in the AST layout to get
    the offset from a pointer to a dynamic object to the base object pointer,
    the base object offset value from the vtable, or the type information
    entry for an object.
    TODO: We should have special operations to do that too.
  }];
}

//===----------------------------------------------------------------------===//
// BoolType
//===----------------------------------------------------------------------===//

def CIR_BoolType : CIR_Type<"Bool", "bool", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
    DeclareTypeInterfaceMethods<CIR_SizedTypeInterface>
]> {
  let summary = "CIR bool type";
  let description = [{
    `!cir.bool` represents C++ bool type.
  }];
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def CIR_ArrayType : CIR_Type<"Array", "array", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
    DeclareTypeInterfaceMethods<CIR_SizedTypeInterface, ["isSized"]>,
]> {
  let summary = "CIR array type";
  let description = [{
    `!cir.array` represents C/C++ constant arrays.
  }];

  let parameters = (ins "mlir::Type":$elementType, "uint64_t":$size);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "mlir::Type":$elementType, "uint64_t":$size
    ), [{
        return $_get(elementType.getContext(), elementType, size);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $elementType `x` $size `>`
  }];

  let extraClassDefinition = [{
    bool $cppClass::isSized() const {
      return ::cir::isSized(getElementType());
    }
  }];
}

//===----------------------------------------------------------------------===//
// VectorType (fixed size)
//===----------------------------------------------------------------------===//

def CIR_VectorType : CIR_Type<"Vector", "vector", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
    DeclareTypeInterfaceMethods<CIR_SizedTypeInterface, ["isSized"]>,
]> {
  let summary = "CIR vector type";
  let description = [{
    The `!cir.vector` type represents a one-dimensional vector.
    It takes three parameters: the element type, the number of elements and the
    scalability flag (optional, defaults to `false`).

    Syntax:

    ```
    vector-type ::= !cir.vector<size x element-type>
    size ::= (decimal-literal | `[` decimal-literal `]`)
    element-type ::= float-type | integer-type | pointer-type
    ```

    The `element-type` must be a scalar CIR type. Zero-sized vectors are not
    allowed. The `size` must be a positive integer.

    Examples:

    ```
    !cir.vector<4 x !cir.int<u, 8>>
    !cir.vector<2 x !cir.float>
    ```

    Scalable vectors are indicated by enclosing size in square brackets.

    Example:
    ```
    !cir.vector<[4] x !cir.int<u, 8>>
    ```
  }];

  let parameters = (ins
    CIR_VectorElementType:$element_type,
    "uint64_t":$size,
    OptionalParameter<"bool">:$is_scalable
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "mlir::Type":$element_type, "uint64_t":$size, CArg<"bool",
      "false">:$is_scalable
    ), [{
        return $_get(element_type.getContext(), element_type, size, is_scalable);
    }]>,
  ];

  let extraClassDefinition = [{
    bool $cppClass::isSized() const {
      return ::cir::isSized(getElementType());
    }
  }];

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// FuncType
//===----------------------------------------------------------------------===//

def CIR_FuncType : CIR_Type<"Func", "func"> {
  let summary = "CIR function type";
  let description = [{
    The `!cir.func` is a function type. It consists of an optional return type,
    a list of parameter types and can optionally be variadic.

    Example:

    ```
    !cir.func<()>
    !cir.func<() -> bool>
    !cir.func<(!s8i, !s8i)>
    !cir.func<(!s8i, !s8i) -> !s32i>
    !cir.func<(!s32i, ...) -> !s32i>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Type">:$inputs,
                        OptionalParameter<"mlir::Type">:$optionalReturnType,
                        "bool":$varArg);
  // Use a custom parser to handle argument types with variadic elipsis.
  let assemblyFormat = [{
    `<` custom<FuncTypeParams>($inputs, $varArg)  (`->` $optionalReturnType^)? `>`
  }];

  let builders = [
    // Create a FuncType, converting the return type from C-style to
    // MLIR-style.  If the given return type is `cir::VoidType`, ignore it
    // and create the FuncType with no return type, which is how MLIR
    // represents function types.
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<mlir::Type>":$inputs, "mlir::Type":$returnType,
      CArg<"bool", "false">:$isVarArg), [{
        return $_get(returnType.getContext(), inputs,
                     mlir::isa<cir::VoidType>(returnType) ? nullptr : returnType,
                     isVarArg);
    }]>
  ];

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns whether the function is variadic.
    bool isVarArg() const { return getVarArg(); }

    /// Returns the `i`th input operand type. Asserts if out of bounds.
    mlir::Type getInput(unsigned i) const { return getInputs()[i]; }

    /// Returns the number of arguments to the function.
    unsigned getNumInputs() const { return getInputs().size(); }

    /// Get the C-style return type of the function, which is !cir.void if the
    /// function returns nothing and the actual return type otherwise.
    mlir::Type getReturnType() const;

    /// Get the MLIR-style return type of the function, which is an empty
    /// ArrayRef if the function returns nothing and a single-element ArrayRef
    /// with the actual return type otherwise.
    llvm::ArrayRef<mlir::Type> getReturnTypes() const;

    /// Does the function type return nothing?
    bool hasVoidReturn() const;

    /// Returns a clone of this function type with the given argument
    /// and result types.
    FuncType clone(mlir::TypeRange inputs, mlir::TypeRange results) const;
  }];
}

//===----------------------------------------------------------------------===//
// MethodType
//===----------------------------------------------------------------------===//

def CIR_MethodType : CIR_Type<"Method", "method", [
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
  DeclareTypeInterfaceMethods<CIR_SizedTypeInterface>
]> {
  let summary = "CIR type that represents C++ pointer-to-member-function type";
  let description = [{
    `cir.method` models the pointer-to-member-function type in C++. The layout
    of this type is ABI-dependent.
  }];

  let parameters = (ins "cir::FuncType":$member_func_ty,
                        "cir::RecordType":$class_ty);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "cir::FuncType":$member_func_ty, "cir::RecordType":$class_ty
    ), [{
      return $_get(member_func_ty.getContext(), member_func_ty, class_ty);
    }]>,
  ];

  let assemblyFormat = [{
    `<` qualified($member_func_ty) `in` $class_ty `>`
  }];
}

//===----------------------------------------------------------------------===//
// Void type
//===----------------------------------------------------------------------===//

def CIR_VoidType : CIR_Type<"Void", "void"> {
  let summary = "CIR void type";
  let description = [{
    The `!cir.void` type represents the C and C++ `void` type.
  }];
  let extraClassDeclaration = [{
    std::string getAlias() const { return "void"; };
  }];
}

//===----------------------------------------------------------------------===//
// StructType
//
// The CIR type for C/C++ struct and class declarations.
//===----------------------------------------------------------------------===//

def CIR_StructType : CIR_Type<"Struct", "struct", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
    DeclareTypeInterfaceMethods<CIR_SizedTypeInterface>,
    MutableType,
]> {
  let summary = "CIR struct/class type";
  let description = [{
    Each unique clang::RecordDecl with struct or class kind is mapped to a
    `cir.struct` type.  Any object in C/C++ that has a struct or class type
    will have a `!cir.struct` in CIR.

    There are three possible formats:

     - Identified and complete: unique name and a known body.
     - Identified and incomplete: unique name and unknown body.
     - Anonymous: no name and a known body.

    Typically, the full data size of a struct object is that which is calculated
    by adding the aligned sizes of all the members. However, note that there is
    one exception to this Flexible Array Members. When the last element is an
    array type of size zero(in CIR), this represents the C/C++ concept of
    'flexible array member', where as long as the type is allocated properly (or
    a global constant), an access through the array member may access outside of
    its bounds. This is necessary to represent C/C++ semantics.

    The optional `class` keyword distinguishes C++ class declarations from
    plain struct declarations.  Both are semantically identical; the keyword
    preserves the original source spelling.

    Examples:

    ```
        !rec_complete = !cir.struct<"complete" {!u8i}>
        !rec_class    = !cir.struct<class "MyClass" {!s32i}>
        !rec_incomplete = !cir.struct<"incomplete" incomplete>
        !anonymous    = !cir.struct<{!u8i}>
        !rec_packed   = !cir.struct<"p1" packed {!u8i, !u8i}>
        !rec_padded   = !cir.struct<"p2" padded {!u8i, !u8i}>
        !recursive    = !cir.struct<"Node" {!cir.ptr<!cir.struct<"Node">>}>
    ```
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"mlir::Type">:$members,
    OptionalParameter<"mlir::StringAttr">:$name,
    "bool":$incomplete,
    "bool":$packed,
    "bool":$padded,
    "bool":$is_class
  );

  // StorageClass is defined in C++ for mutability.
  let storageClass = "StructTypeStorage";
  let genStorageClass = 0;

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;

  let builders = [
    // Create an identified and complete struct/class type.
    TypeBuilder<(ins
      "llvm::ArrayRef<mlir::Type>":$members,
      "mlir::StringAttr":$name,
      "bool":$packed,
      "bool":$padded,
      "bool":$is_class
    ), [{
      return $_get($_ctxt, members, name, /*incomplete=*/false, packed, padded,
                   is_class);
    }]>,

    // Create an identified and incomplete struct/class type.
    TypeBuilder<(ins
      "mlir::StringAttr":$name,
      "bool":$is_class
    ), [{
      return $_get($_ctxt, /*members=*/llvm::ArrayRef<mlir::Type>{}, name,
                   /*incomplete=*/true, /*packed=*/false,
                   /*padded=*/false, is_class);
    }]>,

    // Create an anonymous struct/class type (always complete).
    TypeBuilder<(ins
      "llvm::ArrayRef<mlir::Type>":$members,
      "bool":$packed,
      "bool":$padded,
      "bool":$is_class
    ), [{
      return $_get($_ctxt, members, mlir::StringAttr{}, /*incomplete=*/false,
                   packed, padded, is_class);
    }]>
  ];

  let extraClassDeclaration = [{
    using Base::verifyInvariants;

    bool isClass() const { return getIsClass(); }
    bool isStruct() const { return !getIsClass(); }
    bool isComplete() const { return !isIncomplete(); }
    bool isIncomplete() const;

    size_t getNumElements() const { return getMembers().size(); }
    std::string getKindAsStr() const {
      return getIsClass() ? "class" : "struct";
    }
    mlir::Type getElementType(size_t idx) const {
      return getMembers()[idx];
    }
    std::string getPrefixedName() const {
      return getKindAsStr() + "." + getName().getValue().str();
    }

    void complete(llvm::ArrayRef<mlir::Type> members, bool packed,
                  bool isPadded);

    uint64_t getElementOffset(const mlir::DataLayout &dataLayout,
                              unsigned idx) const;

    bool isLayoutIdentical(const StructType &other);

    // Checks the name of this record to check if it is a 'after' (or during)
    // ABI conversion.
    bool isABIConvertedRecord() const;

    // Get the version of this record's name for use during cxx-abi conversion.
    mlir::StringAttr getABIConvertedName() const;

    // This function should only be called after the CXXABILowering pass. It is
    // intended to do a 'fixup' of the name so that we don't end up with messy
    // looking CIR after CXXABI lowering. This should only be called if all uses
    // of the pre-CXXABILowering version have been removed, otherwise we'll have
    // a conflict in RecordStorage because two types will have the same hash.
    void removeABIConversionNamePrefix();

    bool isSized() const { return isComplete(); }

    /// Returns the data size (excluding tail padding) for this struct type.
    unsigned computeStructDataSize(const mlir::DataLayout &dataLayout) const;

  private:
    static constexpr char abi_conversion_prefix[] = "__post_abi_";
    unsigned computeStructSize(const mlir::DataLayout &dataLayout) const;
    uint64_t computeStructAlignment(const mlir::DataLayout &dataLayout) const;
  public:
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// UnionType
//
// The CIR type for C/C++ union declarations.
//===----------------------------------------------------------------------===//

def CIR_UnionType : CIR_Type<"Union", "union", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
    DeclareTypeInterfaceMethods<CIR_SizedTypeInterface>,
    MutableType,
]> {
  let summary = "CIR union type";
  let description = [{
    Each unique clang::RecordDecl with union kind is mapped to a `cir.union`
    type.  Any object in C/C++ that has a union type will have a `!cir.union`
    in CIR.

    There are three possible formats:

     - Identified and complete: unique name and a known body.
     - Identified and incomplete: unique name and unknown body.
     - Anonymous: no name and a known body.

    Padded unions carry an explicit tail-padding type to ensure the LLVM struct
    that models the union has the correct byte size.

    Examples:

    ```
        !u_complete   = !cir.union<"U" {!s32i, !u8i}>
        !u_incomplete = !cir.union<"U" incomplete>
        !u_anonymous  = !cir.union<{!s32i, !u8i}>
        !u_padded     = !cir.union<"U" {!s32i, !u8i}, padding = {!u8i}>
    ```
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"mlir::Type">:$members,
    OptionalParameter<"mlir::StringAttr">:$name,
    "bool":$incomplete,
    "bool":$packed,
    OptionalParameter<"mlir::Type">:$padding
  );

  // StorageClass is defined in C++ for mutability.
  let storageClass = "UnionTypeStorage";
  let genStorageClass = 0;

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;

  let builders = [
    // Create an identified and complete union type.
    TypeBuilder<(ins
      "llvm::ArrayRef<mlir::Type>":$members,
      "mlir::StringAttr":$name,
      "bool":$packed,
      CArg<"mlir::Type", "{}">:$padding
    ), [{
      return $_get($_ctxt, members, name, /*incomplete=*/false, packed,
                   padding);
    }]>,

    // Create an identified and incomplete union type.
    TypeBuilder<(ins "mlir::StringAttr":$name), [{
      return $_get($_ctxt, /*members=*/llvm::ArrayRef<mlir::Type>{}, name,
                   /*incomplete=*/true, /*packed=*/false,
                   /*padding=*/mlir::Type{});
    }]>,

    // Create an anonymous union type (always complete).
    TypeBuilder<(ins
      "llvm::ArrayRef<mlir::Type>":$members,
      "bool":$packed,
      CArg<"mlir::Type", "{}">:$padding
    ), [{
      return $_get($_ctxt, members, mlir::StringAttr{}, /*incomplete=*/false,
                   packed, padding);
    }]>
  ];

  let extraClassDeclaration = [{
    using Base::verifyInvariants;

    bool isUnion() const { return true; }
    bool isComplete() const { return !isIncomplete(); }
    bool isIncomplete() const;

    /// Returns true when this union carries a tail-padding type.
    /// Derived from getPadding(): a union is padded iff it has a non-null
    /// padding type.
    bool getPadded() const;

    size_t getNumElements() const { return getMembers().size(); }
    std::string getKindAsStr() const { return "union"; }
    mlir::Type getElementType(size_t idx) const {
      return getMembers()[idx];
    }
    std::string getPrefixedName() const {
      return "union." + getName().getValue().str();
    }

    /// Returns the highest-alignment variant of the union (the member used as
    /// the storage type when lowering to LLVM IR).  Returns a null type for
    /// empty unions.
    mlir::Type getUnionStorageType(const mlir::DataLayout &dataLayout) const;
    static mlir::Type getUnionStorageType(const mlir::DataLayout &dataLayout,
                                          llvm::ArrayRef<mlir::Type> members);

    void complete(llvm::ArrayRef<mlir::Type> members, bool packed,
                  mlir::Type padding = {});

    uint64_t getElementOffset(const mlir::DataLayout &, unsigned) const {
      return 0;
    }

    bool isLayoutIdentical(const UnionType &other);

    // Checks the name of this record to check if it is a 'after' (or during)
    // ABI conversion.
    bool isABIConvertedRecord() const;

    // Get the version of this record's name for use during cxx-abi conversion.
    mlir::StringAttr getABIConvertedName() const;

    // This function should only be called after the CXXABILowering pass.
    void removeABIConversionNamePrefix();

    bool isSized() const { return isComplete(); }

  private:
    static constexpr char abi_conversion_prefix[] = "__post_abi_";
  public:
  }];

  let hasCustomAssemblyFormat = 1;
}

// CIRRecordType: tablegen constraint matching either !cir.struct or !cir.union.
// Note: CIRRecordType is used instead of CIR_StructType / CIR_UnionType in op
// definitions so a single constraint covers both record kinds.
def CIRRecordType : Type<
  CPred<"::mlir::isa<::cir::RecordType>($_self)">, "CIR record type">;

//===----------------------------------------------------------------------===//
// Exception Handling Token Types
//===----------------------------------------------------------------------===//

def CIR_EhTokenType : CIR_Type<"EhToken", "eh_token"> {
  let summary = "CIR exception handling token type";
  let description = [{
    `!cir.eh_token` is an opaque type used to track exception handling state
    in flattened CIR. It is returned by `cir.eh.initiate` and passed to
    `cir.eh.dispatch`, `cir.begin_cleanup`, and `cir.begin_catch` operations.

    This token represents an in-flight exception and is used during ABI-lowering
    to generate the appropriate exception handling code.
  }];
}

def CIR_CleanupTokenType : CIR_Type<"CleanupToken", "cleanup_token"> {
  let summary = "CIR cleanup token type";
  let description = [{
    `!cir.cleanup_token` is an opaque type used to track cleanup handling state
    in flattened CIR. It is returned by `cir.begin_cleanup` and consumed by
    `cir.end_cleanup`.

    This token ensures that cleanup regions are properly paired and allows
    the ABI lowering pass to generate appropriate cleanup handling code.
  }];
}

def CIR_CatchTokenType : CIR_Type<"CatchToken", "catch_token"> {
  let summary = "CIR catch token type";
  let description = [{
    `!cir.catch_token` is an opaque type used to track catch handling state
    in flattened CIR. It is returned by `cir.begin_catch` and consumed by
    `cir.end_catch`.

    This token ensures that catch handlers are properly paired and allows
    the ABI lowering pass to generate appropriate exception catching code.
  }];
}

//===----------------------------------------------------------------------===//
// Global type constraints
//===----------------------------------------------------------------------===//

def CIR_AnyType : AnyTypeOf<[
  CIR_VoidType, CIR_BoolType, CIR_ArrayType, CIR_VectorType, CIR_IntType,
  CIR_AnyFloatType, CIR_PointerType, CIR_FuncType, CIR_StructType,
  CIR_UnionType,
  CIR_ComplexType, CIR_VPtrType, CIR_DataMemberType, CIR_MethodType,
  CIR_EhTokenType, CIR_CleanupTokenType, CIR_CatchTokenType
]>;

#endif // CLANG_CIR_DIALECT_IR_CIRTYPES_TD
