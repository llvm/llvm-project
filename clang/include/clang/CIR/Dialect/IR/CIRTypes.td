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
include "clang/CIR/Dialect/IR/CIRTypeConstraints.td"
include "clang/CIR/Interfaces/CIRTypeInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

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
  }];
  let parameters = (ins "unsigned":$width, "bool":$isSigned);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Return true if this is a signed integer type.
    bool isSigned() const { return getIsSigned(); }
    /// Return true if this is an unsigned integer type.
    bool isUnsigned() const { return !getIsSigned(); }
    /// Return type alias.
    std::string getAlias() const {
      return (isSigned() ? 's' : 'u') + std::to_string(getWidth()) + 'i';
    }
    /// Return true if this is a fundamental integer type (i.e. signed or
    /// unsigned integer types whose bit width is 8, 16, 32, or 64).
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
    /// Returns a maximum bitwidth of cir::IntType
    static unsigned maxBitwidth() { return 128; }
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

    ```mlir
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
    The `!cir.ptr` type represents C and C++ pointer types and C++ reference
    types, other than pointers-to-members.  The `pointee` type is the type
    pointed to.

    TODO(CIR): The address space attribute is not yet implemented.
  }];

  let parameters = (ins "mlir::Type":$pointee);

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$pointee), [{
      return $_get(pointee.getContext(), pointee);
    }]>,
    TypeBuilder<(ins "mlir::Type":$pointee), [{
      return $_get($_ctxt, pointee);
    }]>
  ];

  let assemblyFormat = [{
    `<` $pointee  `>`
  }];

  let genVerifyDecl = 1;

  let skipDefaultBuilders = 1;

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
}

//===----------------------------------------------------------------------===//
// CIR_VPtrType
//===----------------------------------------------------------------------===//

def CIR_VPtrType : CIR_Type<"VPtr", "vptr", [
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
    The `!cir.vector` type represents a fixed-size, one-dimensional vector.
    It takes two parameters: the element type and the number of elements.

    Syntax:

    ```mlir
    vector-type ::= !cir.vector<size x element-type>
    element-type ::= float-type | integer-type | pointer-type
    ```

    The `element-type` must be a scalar CIR type. Zero-sized vectors are not
    allowed. The `size` must be a positive integer.

    Examples:

    ```mlir
    !cir.vector<4 x !cir.int<u, 8>>
    !cir.vector<2 x !cir.float>
    ```
  }];

  let parameters = (ins
    CIR_VectorElementType:$elementType,
    "uint64_t":$size
  );

  let assemblyFormat = [{
    `<` $size `x` $elementType `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "mlir::Type":$elementType, "uint64_t":$size
    ), [{
        return $_get(elementType.getContext(), elementType, size);
    }]>,
  ];

  let extraClassDefinition = [{
    bool $cppClass::isSized() const {
      return ::cir::isSized(getElementType());
    }
  }];

  let genVerifyDecl = 1;
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

    ```mlir
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
// RecordType
//
// The base type for all RecordDecls.
//===----------------------------------------------------------------------===//

def CIR_RecordType : CIR_Type<"Record", "record", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
    DeclareTypeInterfaceMethods<CIR_SizedTypeInterface>,
    MutableType,
]> {
  let summary = "CIR record type";
  let description = [{
    Each unique clang::RecordDecl is mapped to a `cir.record` and any object in
    C/C++ that has a struct or class type will have a `cir.record` in CIR.

    There are three possible formats for this type:

     - Identified and complete records: unique name and a known body.
     - Identified and incomplete records: unique name and unknown body.
     - Anonymous records: no name and a known body.

    Identified records are uniqued by their name, and anonymous records are
    uniqued by their body. This means that two anonymous records with the same
    body will be the same type, and two identified records with the same name
    will be the same type. Attempting to build a record with an existing name,
    but a different body will result in an error.

    Each record type will have a `RecordKind` that is either `Class`, `Struct`,
    or `Union`, depending on the C/C++ type that it is representing. Note that
    `Class` and `Struct` are semantically identical, but the kind preserves the
    keyword that was used to declare the type in the original source code.

    A few examples:

    ```mlir
        !rec_complete = !cir.record<struct "complete" {!u8i}>
        !rec_incomplete = !cir.record<struct "incomplete" incomplete>
        !anonymous_struct = !cir.record<struct {!u8i}>
        !rec_p1 = !cir.record<struct "p1" packed {!u8i, !u8i}>
        !rec_p2 = !cir.record<struct "p2" padded {!u8i, !u8i}>
        !rec_p3 = !cir.record<struct "p3" packed padded {!s32i, !u8i, !u8i}>
    ```

    Incomplete records are mutable, meaning they can be later completed with a
    body automatically updating in place every type in the code that uses the
    incomplete record. Mutability allows for recursive types to be represented,
    meaning the record can have members that refer to itself. This is useful for
    representing recursive records and is implemented through a special syntax.
    In the example below, the `Node` record has a member that is a pointer to a
    `Node` record:

    ```mlir
        !s = !cir.record<struct "Node" {!cir.ptr<!cir.record<struct "Node">>}>
    ```
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"mlir::Type">:$members,
    OptionalParameter<"mlir::StringAttr">:$name,
    "bool":$incomplete,
    "bool":$packed,
    "bool":$padded,
    "RecordType::RecordKind":$kind
  );

  // StorageClass is defined in C++ for mutability.
  let storageClass = "RecordTypeStorage";
  let genStorageClass = 0;

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;

  let builders = [
    // Create an identified and complete record type.
    TypeBuilder<(ins
      "llvm::ArrayRef<mlir::Type>":$members,
      "mlir::StringAttr":$name,
      "bool":$packed,
      "bool":$padded,
      "RecordKind":$kind
    ), [{
      return $_get($_ctxt, members, name, /*incomplete=*/false, packed, padded,
                   kind);
    }]>,

    // Create an identified and incomplete record type.
    TypeBuilder<(ins
      "mlir::StringAttr":$name,
      "RecordKind":$kind
    ), [{
      return $_get($_ctxt, /*members=*/llvm::ArrayRef<Type>{}, name,
                         /*incomplete=*/true, /*packed=*/false,
                         /*padded=*/false, kind);
    }]>,

    // Create an anonymous record type (always complete).
    TypeBuilder<(ins
      "llvm::ArrayRef<mlir::Type>":$members,
      "bool":$packed,
      "bool":$padded,
      "RecordKind":$kind
    ), [{
      return $_get($_ctxt, members, mlir::StringAttr{}, /*incomplete=*/false,
                      packed, padded, kind);
    }]>];

  let extraClassDeclaration = [{
    using Base::verifyInvariants;

    enum RecordKind : uint32_t { Class, Struct, Union };

    bool isClass() const { return getKind() == RecordKind::Class; };
    bool isStruct() const { return getKind() == RecordKind::Struct; };
    bool isUnion() const { return getKind() == RecordKind::Union; };
    bool isComplete() const { return !isIncomplete(); };
    bool isIncomplete() const;

    mlir::Type getLargestMember(const mlir::DataLayout &dataLayout) const;
    size_t getNumElements() const { return getMembers().size(); };
    std::string getKindAsStr() {
      switch (getKind()) {
      case RecordKind::Class:
        return "class";
      case RecordKind::Union:
        return "union";
      case RecordKind::Struct:
        return "struct";
      }
      llvm_unreachable("Invalid value for RecordType::getKind()");
    }
    std::string getPrefixedName() {
      return getKindAsStr() + "." + getName().getValue().str();
    }

    void complete(llvm::ArrayRef<mlir::Type> members, bool packed,
                  bool isPadded);

    uint64_t getElementOffset(const mlir::DataLayout &dataLayout,
              unsigned idx) const;

  private:
    unsigned computeStructSize(const mlir::DataLayout &dataLayout) const;
    uint64_t computeStructAlignment(const mlir::DataLayout &dataLayout) const;
  public:
  }];

  let hasCustomAssemblyFormat = 1;
}

// Note CIRRecordType is used instead of CIR_RecordType
// because of tablegen conflicts.
def CIRRecordType : Type<
  CPred<"::mlir::isa<::cir::RecordType>($_self)">, "CIR record type">;

//===----------------------------------------------------------------------===//
// Global type constraints
//===----------------------------------------------------------------------===//

def CIR_AnyType : AnyTypeOf<[
  CIR_VoidType, CIR_BoolType, CIR_ArrayType, CIR_VectorType, CIR_IntType,
  CIR_AnyFloatType, CIR_PointerType, CIR_FuncType, CIR_RecordType,
  CIR_ComplexType, CIR_VPtrType
]>;

#endif // CLANG_CIR_DIALECT_IR_CIRTYPES_TD
