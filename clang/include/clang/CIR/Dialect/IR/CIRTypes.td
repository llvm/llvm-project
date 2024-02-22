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

#ifndef MLIR_CIR_DIALECT_CIR_TYPES
#define MLIR_CIR_DIALECT_CIR_TYPES

include "clang/CIR/Dialect/IR/CIRDialect.td"
include "clang/CIR/Interfaces/ASTAttrInterfaces.td"
include "clang/CIR/Interfaces/CIRFPTypeInterface.td"
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

def CIR_IntType : CIR_Type<"Int", "int",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Integer type with arbitrary precision up to a fixed limit";
  let description = [{
    CIR type that represents C/C++ primitive integer types.
    Said types are: `char`, `short`, `int`, `long`, `long long`, and their \
    unsigned variations.
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
    };
  }];
  let genVerifyDecl = 1;
}

// Constraints

// Unsigned integer type of a specific width.
class UInt<int width>
  : Type<And<[
        CPred<"$_self.isa<::mlir::cir::IntType>()">,
        CPred<"$_self.cast<::mlir::cir::IntType>().isUnsigned()">,
        CPred<"$_self.cast<::mlir::cir::IntType>().getWidth() == " # width>
        ]>, width # "-bit unsigned integer", "::mlir::cir::IntType">,
    BuildableType<
      "mlir::cir::IntType::get($_builder.getContext(), "
      # width # ", /*isSigned=*/false)"> {
  int bitwidth = width;
}

def UInt1  : UInt<1>;
def UInt8  : UInt<8>;
def UInt16 : UInt<16>;
def UInt32 : UInt<32>;
def UInt64 : UInt<64>;

// Signed integer type of a specific width.
class SInt<int width>
  : Type<And<[
        CPred<"$_self.isa<::mlir::cir::IntType>()">,
        CPred<"$_self.cast<::mlir::cir::IntType>().isSigned()">,
        CPred<"$_self.cast<::mlir::cir::IntType>().getWidth() == " # width>
        ]>, width # "-bit signed integer", "::mlir::cir::IntType">,
    BuildableType<
      "mlir::cir::IntType::get($_builder.getContext(), "
      # width # ", /*isSigned=*/true)"> {
  int bitwidth = width;
}

def SInt1  : SInt<1>;
def SInt8  : SInt<8>;
def SInt16 : SInt<16>;
def SInt32 : SInt<32>;
def SInt64 : SInt<64>;

//===----------------------------------------------------------------------===//
// FloatType
//===----------------------------------------------------------------------===//

class CIR_FloatType<string name, string mnemonic> 
    : CIR_Type<name, mnemonic,
          [
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
            DeclareTypeInterfaceMethods<CIRFPTypeInterface>,
          ]> {}

def CIR_Single : CIR_FloatType<"Single", "float"> {
  let summary = "CIR single-precision float type";
  let description = [{
    Floating-point type that represents the `float` type in C/C++. Its
    underlying floating-point format is the IEEE-754 binary32 format.
  }];
}

def CIR_Double : CIR_FloatType<"Double", "double"> {
  let summary = "CIR double-precision float type";
  let description = [{
    Floating-point type that represents the `double` type in C/C++. Its
    underlying floating-point format is the IEEE-754 binar64 format.
  }];
}

// Constraints

def CIR_AnyFloat: AnyTypeOf<[CIR_Single, CIR_Double]>;

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

def CIR_PointerType : CIR_Type<"Pointer", "ptr",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR pointer type";
  let description = [{
    `CIR.ptr` is a type returned by any op generating a pointer in C++.
  }];

  let parameters = (ins "mlir::Type":$pointee);

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// DataMemberType
//===----------------------------------------------------------------------===//

def CIR_DataMemberType : CIR_Type<"DataMember", "data_member",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR type that represents pointer-to-data-member type in C++";
  let description = [{
    `cir.member_ptr` models the pointer-to-data-member type in C++. Values of
    this type are essentially offsets of the pointed-to member within one of
    its containing struct.
  }];

  let parameters = (ins "mlir::Type":$memberTy,
                        "mlir::cir::StructType":$clsTy);

  let assemblyFormat = [{
    `<` $memberTy `in` $clsTy `>`
  }];
}

//===----------------------------------------------------------------------===//
// BoolType
//
// An alternative here is to represent bool as mlir::i1, but let's be more
// generic.
//
//===----------------------------------------------------------------------===//

def CIR_BoolType :
    CIR_Type<"Bool", "bool",
             [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR bool type";
  let description = [{
    `cir.bool` represent's C++ bool type.
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def CIR_ArrayType : CIR_Type<"Array", "array",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR array type";
  let description = [{
    `CIR.array` represents C/C++ constant arrays.
  }];

  let parameters = (ins "mlir::Type":$eltType, "uint64_t":$size);

  let assemblyFormat = [{
    `<` $eltType `x` $size `>`
  }];
}

//===----------------------------------------------------------------------===//
// VectorType (fixed size)
//===----------------------------------------------------------------------===//

def CIR_VectorType : CIR_Type<"Vector", "vector",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR vector type";
  let description = [{
    `cir.vector' represents fixed-size vector types.  The parameters are the
    element type and the number of elements.
  }];

  let parameters = (ins "mlir::Type":$eltType, "uint64_t":$size);

  let assemblyFormat = [{
    `<` $eltType `x` $size `>`
  }];
}

//===----------------------------------------------------------------------===//
// FuncType
//===----------------------------------------------------------------------===//

def CIR_FuncType : CIR_Type<"Func", "func"> {
  let summary = "CIR function type";
  let description = [{
    The `!cir.func` is a function type. It consists of a single return type, a
    list of parameter types and can optionally be variadic.

    Example:

    ```mlir
    !cir.func<!bool ()>
    !cir.func<!s32i (!s8i, !s8i)>
    !cir.func<!s32i (!s32i, ...)>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"Type">:$inputs, "Type":$returnType,
                        "bool":$varArg);
  let assemblyFormat = [{
    `<` $returnType ` ` `(` custom<FuncTypeArgs>($inputs, $varArg) `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<Type>":$inputs, "Type":$returnType,
      CArg<"bool", "false">:$isVarArg), [{
      return $_get(returnType.getContext(), inputs, returnType, isVarArg);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns whether the function is variadic.
    bool isVarArg() const { return getVarArg(); }

    /// Returns the `i`th input operand type. Asserts if out of bounds.
    Type getInput(unsigned i) const { return getInputs()[i]; }

    /// Returns the number of arguments to the function.
    unsigned getNumInputs() const { return getInputs().size(); }

    /// Returns the result type of the function as an ArrayRef, enabling better
    /// integration with generic MLIR utilities.
    ArrayRef<Type> getReturnTypes() const;

    /// Returns whether the function is returns void.
    bool isVoid() const;

    /// Returns a clone of this function type with the given argument
    /// and result types.
    FuncType clone(TypeRange inputs, TypeRange results) const;
  }];
}

//===----------------------------------------------------------------------===//
// Exception info type
//
// By introducing an exception info type, exception related operations can be
// more descriptive.
//
// This basically wraps a uint8_t* and a uint32_t
//
//===----------------------------------------------------------------------===//

def CIR_ExceptionInfo : CIR_Type<"ExceptionInfo", "eh.info"> {
  let summary = "CIR exception info";
  let description = [{
    Represents the content necessary for a `cir.call` to pass back an exception
    object pointer + some extra selector information. This type is required for
    some exception related operations, like `cir.catch`, `cir.eh.selector_slot`
    and `cir.eh.slot`.
  }];
}

//===----------------------------------------------------------------------===//
// Void type
//===----------------------------------------------------------------------===//

def CIR_VoidType : CIR_Type<"Void", "void"> {
  let summary = "CIR void type";
  let description = [{
    The `!cir.void` type represents the C/C++ `void` type.
  }];
  let extraClassDeclaration = [{
    /// Returns a clone of this type with the given context.
    std::string getAlias() const { return "void"; };
  }];
}

// Constraints

// Pointer to void
def VoidPtr : Type<
    And<[
      CPred<"$_self.isa<::mlir::cir::PointerType>()">,
      CPred<"$_self.cast<::mlir::cir::PointerType>()"
            ".getPointee().isa<::mlir::cir::VoidType>()">,
    ]>, "void*">,
    BuildableType<
      "mlir::cir::PointerType::get($_builder.getContext(),"
      "mlir::cir::VoidType::get($_builder.getContext()))"> {
}

// Pointer to struct
def StructPtr : Type<
    And<[
      CPred<"$_self.isa<::mlir::cir::PointerType>()">,
      CPred<"$_self.cast<::mlir::cir::PointerType>()"
            ".getPointee().isa<::mlir::cir::StructType>()">,
    ]>, "!cir.struct*"> {
}

// Pointers to exception info
def ExceptionInfoPtr : Type<
    And<[
      CPred<"$_self.isa<::mlir::cir::PointerType>()">,
      CPred<"$_self.cast<::mlir::cir::PointerType>()"
            ".getPointee().isa<::mlir::cir::ExceptionInfoType>()">,
    ]>, "!cir.eh_info*">,
    BuildableType<
      "mlir::cir::PointerType::get($_builder.getContext(),"
      "mlir::cir::ExceptionInfoType::get($_builder.getContext()))"> {
}

// Pooint to pointers to exception info
def ExceptionInfoPtrPtr : Type<
    And<[
      CPred<"$_self.isa<::mlir::cir::PointerType>()">,
      And<[
        CPred<"$_self.cast<::mlir::cir::PointerType>()"
              ".getPointee().isa<::mlir::cir::PointerType>()">,
        CPred<"$_self.cast<::mlir::cir::PointerType>()"
              ".getPointee().cast<::mlir::cir::PointerType>()"
              ".getPointee().isa<::mlir::cir::ExceptionInfoType>()">,
      ]>
    ]>, "!cir.eh_info**">,
    BuildableType<
      "mlir::cir::PointerType::get($_builder.getContext(),"
      "mlir::cir::PointerType::get($_builder.getContext(),"
      "mlir::cir::ExceptionInfoType::get($_builder.getContext())))"> {
}

//===----------------------------------------------------------------------===//
// StructType (defined in cpp files)
//===----------------------------------------------------------------------===//

def CIR_StructType : Type<CPred<"$_self.isa<::mlir::cir::StructType>()">,
                          "CIR struct type">;

//===----------------------------------------------------------------------===//
// Global type constraints
//===----------------------------------------------------------------------===//

def CIR_AnyType : AnyTypeOf<[
  CIR_IntType, CIR_PointerType, CIR_DataMemberType, CIR_BoolType, CIR_ArrayType,
  CIR_VectorType, CIR_FuncType, CIR_VoidType, CIR_StructType, CIR_ExceptionInfo,
  CIR_AnyFloat,
]>;

#endif // MLIR_CIR_DIALECT_CIR_TYPES
