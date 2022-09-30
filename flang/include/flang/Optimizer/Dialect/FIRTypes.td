//===- FIRTypes.td - FIR types -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the FIR dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef FIR_DIALECT_FIR_TYPES
#define FIR_DIALECT_FIR_TYPES

include "mlir/IR/AttrTypeBase.td"
include "flang/Optimizer/Dialect/FIRDialect.td"

//===----------------------------------------------------------------------===//
// FIR Types
//===----------------------------------------------------------------------===//

class FIR_Type<string name, string typeMnemonic> : TypeDef<fir_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def fir_Type : Type<CPred<"fir::isa_fir_or_std_type($_self)">,
    "FIR dialect type">;

def fir_BoxCharType : FIR_Type<"BoxChar", "boxchar"> {
  let summary = "CHARACTER type descriptor.";

  let description = [{
    The type of a pair that describes a CHARACTER variable. Specifically, a
    CHARACTER consists of a reference to a buffer (the string value) and a LEN
    type parameter (the runtime length of the buffer).
  }];

  let parameters = (ins "KindTy":$kind);

  let genAccessors = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    using KindTy = unsigned;

    // a !fir.boxchar<k> always wraps a !fir.char<k, ?>
    CharacterType getElementType(mlir::MLIRContext *context) const;

    CharacterType getEleTy() const;
  }];
}

def fir_BoxProcType : FIR_Type<"BoxProc", "boxproc"> {
  let summary = "";

  let description = [{
    The type of a pair that describes a PROCEDURE reference. Pointers to
    internal procedures must carry an additional reference to the host's
    variables that are referenced.
  }];

  let parameters = (ins "mlir::Type":$eleTy);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def fir_BoxType : FIR_Type<"Box", "box"> {
  let summary = "The type of a Fortran descriptor";

  let description = [{
    Descriptors are tuples of information that describe an entity being passed
    from a calling context. This information might include (but is not limited
    to) whether the entity is an array, its size, or what type it has.
  }];

  let parameters = (ins "mlir::Type":$eleTy);

  let skipDefaultBuilders = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
        "mlir::Type":$eleTy), [{
      return Base::get(eleTy.getContext(), eleTy);
    }]>,
  ];

  let extraClassDeclaration = [{
    mlir::Type getElementType() const { return getEleTy(); }
  }];

  let genVerifyDecl = 1;

  let assemblyFormat = "`<` $eleTy `>`";
}

def fir_CharacterType : FIR_Type<"Character", "char"> {
  let summary = "FIR character type";

  let description = [{
    Model of the Fortran CHARACTER intrinsic type, including the KIND type
    parameter. The model optionally includes a LEN type parameter. A
    CharacterType is thus the type of both a single character value and a
    character with a LEN parameter.
  }];

  let parameters = (ins "KindTy":$FKind, "CharacterType::LenType":$len);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    using KindTy = unsigned;
    using LenType = std::int64_t;

    /// Return unknown length Character type. e.g., CHARACTER(LEN=n).
    static CharacterType getUnknownLen(mlir::MLIRContext *ctxt, KindTy kind) {
      return get(ctxt, kind, unknownLen());
    }

    /// Return length 1 Character type. e.g., CHARACTER(LEN=1).
    static CharacterType getSingleton(mlir::MLIRContext *ctxt, KindTy kind) {
      return get(ctxt, kind, singleton());
    }

    /// Character is a singleton and has a LEN of 1.
    static constexpr LenType singleton() { return 1; }

    /// Character has a LEN value which is not a compile-time known constant.
    static constexpr LenType unknownLen() { return -1; }

    /// Character LEN is a runtime value.
    bool hasDynamicLen() { return getLen() == unknownLen(); }

    /// Character LEN is a compile-time constant value.
    bool hasConstantLen() { return !hasDynamicLen(); }
  }];
}

def fir_ComplexType : FIR_Type<"Complex", "complex"> {
  let summary = "Complex type";

  let description = [{
    Model of a Fortran COMPLEX intrinsic type, including the KIND type
    parameter. COMPLEX is a floating point type with a real and imaginary
    member.
  }];

  let parameters = (ins "KindTy":$fKind);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    using KindTy = unsigned;

    mlir::Type getElementType() const;
    mlir::Type getEleType(const fir::KindMapping &kindMap) const;
  }];
}

def fir_FieldType : FIR_Type<"Field", "field"> {
  let summary = "A field (in a RecordType) argument's type";

  let description = [{
    The type of a field name. Implementations may defer the layout of a Fortran
    derived type until runtime. This implies that the runtime must be able to
    determine the offset of fields within the entity.
  }];
}

def fir_HeapType : FIR_Type<"Heap", "heap"> {
  let summary = "Reference to an ALLOCATABLE attribute type";

  let description = [{
    The type of a heap pointer. Fortran entities with the ALLOCATABLE attribute
    may be allocated on the heap at runtime. These pointers are explicitly
    distinguished to disallow the composition of multiple levels of
    indirection. For example, an ALLOCATABLE POINTER is invalid.
  }];

  let parameters = (ins "mlir::Type":$eleTy);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType), [{
      assert(singleIndirectionLevel(elementType) && "invalid element type");
      return Base::get(elementType.getContext(), elementType);
    }]>,
  ];
}

def fir_IntegerType : FIR_Type<"Integer", "int"> {
  let summary = "FIR integer type";

  let description = [{
    Model of a Fortran INTEGER intrinsic type, including the KIND type
    parameter.
  }];

  let parameters = (ins "KindTy":$fKind);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    using KindTy = unsigned;
  }];
}

def fir_LenType : FIR_Type<"Len", "len"> {
  let summary = "A LEN parameter (in a RecordType) argument's type";

  let description = [{
    The type of a LEN parameter name. Implementations may defer the layout of a
    Fortran derived type until runtime. This implies that the runtime must be
    able to determine the offset of LEN type parameters related to an entity.
  }];
}

def fir_LogicalType : FIR_Type<"Logical", "logical"> {
  let summary = "FIR logical type";

  let description = [{
    Model of a Fortran LOGICAL intrinsic type, including the KIND type
    parameter.
  }];

  let parameters = (ins "KindTy":$fKind);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    using KindTy = unsigned;
  }];
}

def fir_LLVMPointerType : FIR_Type<"LLVMPointer", "llvm_ptr"> {
  let summary = "Like LLVM pointer type";

  let description = [{
    A pointer type that does not have any of the constraints and semantics
    of other FIR pointer types and that translates to llvm pointer types.
    It is meant to implement indirection that cannot be expressed directly
    in Fortran, but are needed to implement some Fortran features (e.g,
    double indirections).
  }];

  let parameters = (ins "mlir::Type":$eleTy);

  let assemblyFormat = "`<` $eleTy `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>,
  ];
}

def fir_PointerType : FIR_Type<"Pointer", "ptr"> {
  let summary = "Reference to a POINTER attribute type";

  let description = [{
    The type of entities with the POINTER attribute.  These pointers are
    explicitly distinguished to disallow the composition of multiple levels of
    indirection. For example, an ALLOCATABLE POINTER is invalid.
  }];

  let parameters = (ins "mlir::Type":$eleTy);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType), [{
      assert(singleIndirectionLevel(elementType) && "invalid element type");
      return Base::get(elementType.getContext(), elementType);
    }]>,
  ];

  let extraClassDeclaration = [{
    mlir::Type getElementType() const { return getEleTy(); }
  }];
}

def fir_RealType : FIR_Type<"Real", "real"> {
  let summary = "FIR real type";

  let description = [{
    Model of a Fortran REAL (and DOUBLE PRECISION) intrinsic type, including the
    KIND type parameter.
  }];

  let parameters = (ins "KindTy":$fKind);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    using KindTy = unsigned;
  }];

  let genVerifyDecl = 1;
}

def fir_RecordType : FIR_Type<"Record", "type"> {
  let summary = "FIR derived type";

  let description = [{
    Model of Fortran's derived type, TYPE. The name of the TYPE includes any
    KIND type parameters. The record includes runtime slots for LEN type
    parameters and for data components.
  }];

  let parameters = (ins StringRefParameter<"name">:$name);

  let genVerifyDecl = 1;
  let genStorageClass = 0;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    using TypePair = std::pair<std::string, mlir::Type>;
    using TypeList = std::vector<TypePair>;
    TypeList getTypeList() const;
    TypeList getLenParamList() const;

    mlir::Type getType(llvm::StringRef ident);
    // Returns the index of the field \p ident in the type list.
    // Returns maximum unsigned if ident is not a field of this RecordType.
    unsigned getFieldIndex(llvm::StringRef ident);
    mlir::Type getType(unsigned index) {
      assert(index < getNumFields());
      return getTypeList()[index].second;
    }
    unsigned getNumFields() { return getTypeList().size(); }
    unsigned getNumLenParams() { return getLenParamList().size(); }
    bool isDependentType() { return getNumLenParams(); }

    void finalize(llvm::ArrayRef<TypePair> lenPList,
                  llvm::ArrayRef<TypePair> typeList);

    detail::RecordTypeStorage const *uniqueKey() const;
  }];
}

def fir_ReferenceType : FIR_Type<"Reference", "ref"> {
  let summary = "Reference to an entity type";

  let description = [{
    The type of a reference to an entity in memory.
  }];

  let parameters = (ins "mlir::Type":$eleTy);

  let skipDefaultBuilders = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>,
  ];

  let extraClassDeclaration = [{
    mlir::Type getElementType() const { return getEleTy(); }
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def fir_ShapeType : FIR_Type<"Shape", "shape"> {
  let summary = "shape of a multidimensional array object";

  let description = [{
    Type of a vector of runtime values that define the shape of a
    multidimensional array object. The vector is the extents of each array
    dimension. The rank of a ShapeType must be at least 1.
  }];

  let parameters = (ins "unsigned":$rank);
  let hasCustomAssemblyFormat = 1;
}

def fir_ShapeShiftType : FIR_Type<"ShapeShift", "shapeshift"> {
  let summary = "shape and origin of a multidimensional array object";

  let description = [{
    Type of a vector of runtime values that define the shape and the origin of a
    multidimensional array object. The vector is of pairs, origin offset and
    extent, of each array dimension. The rank of a ShapeShiftType must be at
    least 1.
  }];

  let parameters = (ins "unsigned":$rank);
  let hasCustomAssemblyFormat = 1;
}

def fir_ShiftType : FIR_Type<"Shift", "shift"> {
  let summary = "lower bounds of a multidimensional array object";

  let description = [{
    Type of a vector of runtime values that define the lower bounds of a
    multidimensional array object. The vector is the lower bounds of each array
    dimension. The rank of a ShiftType must be at least 1.
  }];

  let parameters = (ins "unsigned":$rank);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    using KindTy = unsigned;

    // a !fir.boxchar<k> always wraps a !fir.char<k, ?>
    CharacterType getElementType(mlir::MLIRContext *context) const;

    CharacterType getEleTy() const;
  }];
}

def fir_SequenceType : FIR_Type<"Sequence", "array"> {
  let summary = "FIR array type";

  let description = [{
    A sequence type is a multi-dimensional array of values. The sequence type
    may have an unknown number of dimensions or the extent of dimensions may be
    unknown. A sequence type models a Fortran array entity, giving it a type in
    FIR. A sequence type is assumed to be stored in a column-major order, which
    differs from LLVM IR and other dialects of MLIR.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "Sequence shape">:$shape,
    "mlir::Type":$eleTy,
    "mlir::AffineMapAttr":$layoutMap
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
        "llvm::ArrayRef<int64_t>":$shape,
        "mlir::Type":$eleTy), [{
      return get(eleTy.getContext(), shape, eleTy, {});
    }]>,
    TypeBuilderWithInferredContext<(ins
        "mlir::Type":$eleTy,
        "size_t":$dimensions), [{
      llvm::SmallVector<int64_t> shape(dimensions, getUnknownExtent());
      return get(eleTy.getContext(), shape, eleTy, {});
    }]>
  ];

  let extraClassDeclaration = [{
    using Extent = int64_t;
    using Shape = llvm::SmallVector<Extent>;
    using ShapeRef = llvm::ArrayRef<int64_t>;
    unsigned getConstantRows() const;

    // The number of dimensions of the sequence
    unsigned getDimension() const { return getShape().size(); }

    // Is the shape of the sequence dynamic?
    bool hasDynamicExtents() const {
      for(const auto d : getShape())
        if (d == getUnknownExtent())
          return true;
      return false;
    }

    // Does the sequence have unknown shape? (`array<* x T>`)
    bool hasUnknownShape() const { return getShape().empty(); }

    // The value `-1` represents an unknown extent for a dimension
    static constexpr Extent getUnknownExtent() { return -1; }
  }];
}

def fir_SliceType : FIR_Type<"Slice", "slice"> {
  let summary = "FIR slice";

  let description = [{
    Type of a vector that represents an array slice operation on an array.
    Fortran slices are triples of lower bound, upper bound, and stride. The rank
    of a SliceType must be at least 1.
  }];

  let parameters = (ins "unsigned":$rank);
  let hasCustomAssemblyFormat = 1;
}

def fir_TypeDescType : FIR_Type<"TypeDesc", "tdesc"> {
  let summary = "FIR Type descriptor type";

  let description = [{
    The type of a type descriptor object. The runtime may generate type
    descriptor objects to determine the type of an entity at runtime, etc.
  }];

  let parameters = (ins "mlir::Type":$ofTy);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>,
  ];
}

def fir_VectorType : FIR_Type<"Vector", "vector"> {
  let summary = "FIR vector type";

  let description = [{
    Replacement for the builtin vector type.
    The FIR vector type is always rank one. It's size is always a constant.
    A vector's element type must be real or integer.
  }];

  let parameters = (ins "uint64_t":$len, "mlir::Type":$eleTy);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static bool isValidElementType(mlir::Type t);
  }];

  let skipDefaultBuilders = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
        "uint64_t":$len,
        "mlir::Type":$eleTy), [{
      return Base::get(eleTy.getContext(), len, eleTy);
    }]>,
  ];
}

def fir_VoidType : FIR_Type<"Void", "void"> {
  let genStorageClass = 0;
}

// Generalized FIR and standard dialect types representing intrinsic types
def AnyIntegerLike : TypeConstraint<Or<[SignlessIntegerLike.predicate,
    AnySignedInteger.predicate, fir_IntegerType.predicate]>, "any integer">;
def AnyLogicalLike : TypeConstraint<Or<[BoolLike.predicate,
    fir_LogicalType.predicate]>, "any logical">;
def AnyRealLike : TypeConstraint<Or<[FloatLike.predicate,
    fir_RealType.predicate]>, "any real">;
def AnyIntegerType : Type<AnyIntegerLike.predicate, "any integer">;

// Composable types
// Note that we include both fir_ComplexType and AnyComplex, so we can use both
// the FIR ComplexType and the MLIR ComplexType (the former is used to represent
// Fortran complex and the latter for C++ std::complex).
def AnyCompositeLike : TypeConstraint<Or<[fir_RecordType.predicate,
    fir_SequenceType.predicate, fir_ComplexType.predicate, AnyComplex.predicate,
    fir_VectorType.predicate, IsTupleTypePred, fir_CharacterType.predicate]>,
    "any composite">;

// Reference types
def AnyReferenceLike : TypeConstraint<Or<[fir_ReferenceType.predicate,
    fir_HeapType.predicate, fir_PointerType.predicate,
    fir_LLVMPointerType.predicate]>, "any reference">;

def FuncType : TypeConstraint<FunctionType.predicate, "function type">;

def AnyCodeOrDataRefLike : TypeConstraint<Or<[AnyReferenceLike.predicate,
    FunctionType.predicate]>, "any code or data reference">;

def RefOrLLVMPtr : TypeConstraint<Or<[fir_ReferenceType.predicate,
    fir_LLVMPointerType.predicate]>, "fir.ref or fir.llvm_ptr">;

def AnyBoxLike : TypeConstraint<Or<[fir_BoxType.predicate,
    fir_BoxCharType.predicate, fir_BoxProcType.predicate]>, "any box">;

def AnyRefOrBoxLike : TypeConstraint<Or<[AnyReferenceLike.predicate,
    AnyBoxLike.predicate, FunctionType.predicate]>,
    "any reference or box like">;
def AnyRefOrBox : TypeConstraint<Or<[fir_ReferenceType.predicate,
    fir_HeapType.predicate, fir_PointerType.predicate, fir_BoxType.predicate]>,
    "any reference or box">;

def AnyShapeLike : TypeConstraint<Or<[fir_ShapeType.predicate,
    fir_ShapeShiftType.predicate]>, "any legal shape type">;
def AnyShapeType : Type<AnyShapeLike.predicate, "any legal shape type">;
def AnyShapeOrShiftLike : TypeConstraint<Or<[fir_ShapeType.predicate,
    fir_ShapeShiftType.predicate, fir_ShiftType.predicate]>,
    "any legal shape or shift type">;
def AnyShapeOrShiftType : Type<AnyShapeOrShiftLike.predicate,
    "any legal shape or shift type">;

def AnyEmboxLike : TypeConstraint<Or<[AnySignlessInteger.predicate,
    Index.predicate, fir_IntegerType.predicate]>,
    "any legal embox argument type">;
def AnyEmboxArg : Type<AnyEmboxLike.predicate, "embox argument type">;

def AnyComponentLike : TypeConstraint<Or<[AnySignlessInteger.predicate,
    Index.predicate, fir_IntegerType.predicate, fir_FieldType.predicate]>,
    "any coordinate index">;
def AnyComponentType : Type<AnyComponentLike.predicate, "coordinate type">;

def AnyCoordinateLike : TypeConstraint<Or<[AnySignlessInteger.predicate,
    Index.predicate, fir_IntegerType.predicate, fir_FieldType.predicate,
    fir_LenType.predicate]>, "any coordinate index">;
def AnyCoordinateType : Type<AnyCoordinateLike.predicate, "coordinate type">;

// The legal types of global symbols
def AnyAddressableLike : TypeConstraint<Or<[fir_ReferenceType.predicate,
    FunctionType.predicate]>, "any addressable">;

def ArrayOrBoxOrRecord : TypeConstraint<Or<[fir_SequenceType.predicate,
    fir_BoxType.predicate, fir_RecordType.predicate]>,
    "fir.box, fir.array or fir.type">;


#endif // FIR_DIALECT_FIR_TYPES
