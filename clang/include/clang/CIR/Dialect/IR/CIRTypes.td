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
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// CIR Types
//===----------------------------------------------------------------------===//

class CIR_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<CIR_Dialect, name, traits> {
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
// StructType
//
// The base type for all RecordDecls.
//
//===----------------------------------------------------------------------===//

def CIR_StructType : CIR_Type<"Struct", "struct",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR struct type";
  let description = [{
    Each unique clang::RecordDecl is mapped to a `cir.struct` and any object in
    C/C++ that has a struct type will have a `cir.struct` in CIR.
  }];

  let parameters = (ins
    ArrayRefParameter<"mlir::Type", "members">:$members,
    "mlir::StringAttr":$name,
    "bool":$incomplete,
    "bool":$packed,
    "mlir::cir::StructType::RecordKind":$kind,
    "ASTRecordDeclInterface":$ast
  );

  let skipDefaultBuilders = 1;
  let builders = [
    // Build an identified and complete struct.
    TypeBuilder<(ins
      "ArrayRef<Type>":$members,
      "StringAttr":$name,
      "bool":$packed,
      "RecordKind":$kind,
      CArg<"ASTRecordDeclInterface", "nullptr">:$ast), [{
      return $_get(context, members, name, /*incomplete=*/false,
                   packed, kind, ast);
    }]>,
    // Build an incomplete struct.
    TypeBuilder<(ins
      "StringAttr":$name,
      "RecordKind":$kind), [{
      return $_get(context, /*members=*/ArrayRef<Type>{}, name,
                   /*incomplete=*/true, /*packed=*/false, kind,
                   /*ast=*/nullptr);
    }]>,
    // Build an anonymous struct.
    TypeBuilder<(ins
      "ArrayRef<Type>":$members,
      "bool":$packed,
      "RecordKind":$kind,
      CArg<"ASTRecordDeclInterface", "nullptr">:$ast), [{
      return $_get(context, members, /*name=*/nullptr,
                   /*incomplete=*/false, packed, kind, ast);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    enum RecordKind : uint32_t {
      Class,
      Union,
      Struct
    };

  private:
    // All these support lazily computation and storage
    // for the struct size and alignment.
    mutable std::optional<unsigned> size{}, align{};
    mutable std::optional<bool> padded{};
    mutable mlir::Type largestMember{};
    void computeSizeAndAlignment(const ::mlir::DataLayout &dataLayout) const;
  public:
    void dropAst();
    size_t getNumElements() const { return getMembers().size(); }
    bool isIncomplete() const { return getIncomplete(); }
    bool isComplete() const { return !getIncomplete(); }
    bool isPadded(const ::mlir::DataLayout &dataLayout) const;

    std::string getKindAsStr() {
      switch (getKind()) {
      case RecordKind::Class:
        return "class";
      case RecordKind::Union:
        return "union";
      case RecordKind::Struct:
        return "struct";
      }
    }

    std::string getPrefixedName() {
      return getKindAsStr() + "." + getName().getValue().str();
    }

    /// Return the member with the largest bit-length.
    mlir::Type getLargestMember(const ::mlir::DataLayout &dataLayout) const;

    /// Return whether this is a class declaration.
    bool isClass() const { return getKind() == RecordKind::Class; }

    /// Return whether this is a union declaration.
    bool isUnion() const { return getKind() == RecordKind::Union; }

    /// Return whether this is a struct declaration.
    bool isStruct() const { return getKind() == RecordKind::Struct; }
  }];

  let extraClassDefinition = [{
    void $cppClass::dropAst() {
      getImpl()->ast = nullptr;
    }
  }];
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

//===----------------------------------------------------------------------===//
// One type to bind them all
//===----------------------------------------------------------------------===//

def CIR_AnyCIRType : AnyTypeOf<[CIR_PointerType, CIR_BoolType, CIR_StructType,
                                CIR_ArrayType, CIR_FuncType, CIR_VoidType]>;

#endif // MLIR_CIR_DIALECT_CIR_TYPES
