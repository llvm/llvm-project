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

//===----------------------------------------------------------------------===//
// FIR Types
//===----------------------------------------------------------------------===//

class FIR_Type<string name, string typeMnemonic> : TypeDef<fir_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def fir_BoxProcType : FIR_Type<"BoxProc", "boxproc"> {
  let summary = "";

  let description = [{
    The type of a pair that describes a PROCEDURE reference. Pointers to
    internal procedures must carry an additional reference to the host's
    variables that are referenced.
  }];

  let parameters = (ins "mlir::Type":$eleTy);

  let printer = [{
    $_printer << "boxproc<";
    $_printer.printType(getEleTy());
    $_printer << '>';
  }];

  let genAccessors = 1;

  let genVerifyInvariantsDecl = 1;
}

def BoxType : FIR_Type<"Box", "box"> {
  let summary = "The type of a Fortran descriptor";

  let description = [{
    Descriptors are tuples of information that describe an entity being passed
    from a calling context. This information might include (but is not limited
    to) whether the entity is an array, its size, or what type it has.
  }];

  let parameters = (ins "mlir::Type":$eleTy, "mlir::AffineMapAttr":$map);

  let extraClassDeclaration = [{
    mlir::Type getElementType() const { return getEleTy(); }
    mlir::AffineMapAttr getLayoutMap() const { return getMap(); }
    static BoxType get(mlir::Type eleTy, mlir::AffineMapAttr map = {}) {
      return get(eleTy.getContext(), eleTy, map);
    }
  }];

  let genAccessors = 1;

  let genVerifyInvariantsDecl = 1;
}

def fir_FieldType : FIR_Type<"Field", "field"> {
  let summary = "A field (in a RecordType) argument's type";

  let description = [{
    The type of a field name. Implementations may defer the layout of a Fortran
    derived type until runtime. This implies that the runtime must be able to
    determine the offset of fields within the entity.
  }];

  let printer = [{
    $_printer << "field";
  }];

  let parser = [{
    return get(context);
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

  let printer = [{
    $_printer << "complex<" << getFKind() << '>';
  }];

  let genAccessors = 1;

  let extraClassDeclaration = [{
    using KindTy = unsigned;

    mlir::Type getElementType() const;
  }];
}

def ShapeType : FIR_Type<"Shape", "shape"> {
  let summary = "shape of a multidimensional array object";

  let description = [{
    Type of a vector of runtime values that define the shape of a
    multidimensional array object. The vector is the extents of each array
    dimension. The rank of a ShapeType must be at least 1.
  }];

  let parameters = (ins "unsigned":$rank);

  let printer = [{
    $_printer << "shape<" << getImpl()->rank << ">";
  }];

  let parser = [{
    int rank;
    if ($_parser.parseLess() || $_parser.parseInteger(rank) ||
        $_parser.parseGreater())
      return Type();
    return get(context, rank);
  }];
}

def ShapeShiftType : FIR_Type<"ShapeShift", "shapeshift"> {
  let summary = "shape and origin of a multidimensional array object";

  let description = [{
    Type of a vector of runtime values that define the shape and the origin of a
    multidimensional array object. The vector is of pairs, origin offset and
    extent, of each array dimension. The rank of a ShapeShiftType must be at
    least 1.
  }];

  let parameters = (ins "unsigned":$rank);

  let printer = [{
    $_printer << "shapeshift<" << getImpl()->rank << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    int rank;
    if ($_parser.parseInteger(rank))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get(context, rank);
  }];
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

  let extraClassDeclaration = [{
    using KindTy = unsigned;
    using LenType = std::int64_t;

    // Return unknown length CHARACTER type.
    static CharacterType getUnknownLen(mlir::MLIRContext *ctxt, KindTy kind) {
      return get(ctxt, kind, unknownLen());
    }

    // Return length 1 CHARACTER type.
    static CharacterType getSingleton(mlir::MLIRContext *ctxt, KindTy kind) {
      return get(ctxt, kind, singleton());
    }

    // CHARACTER is a singleton and has a LEN of 1.
    static constexpr LenType singleton() { return 1; }
    // CHARACTER has an unknown LEN property.
    static constexpr LenType unknownLen() { return -1; }
  }];
}

def fir_BoxCharType : FIR_Type<"BoxChar", "boxchar"> {
  let summary = "CHARACTER type descriptor.";

  let description = [{
    The type of a pair that describes a CHARACTER variable. Specifically, a
    CHARACTER consists of a reference to a buffer (the string value) and a LEN
    type parameter (the runtime length of the buffer).
  }];

  let parameters = (ins "KindTy":$kind);

  let printer = [{
    $_printer << "boxchar<" << getImpl()->kind << ">";
  }];

  let genAccessors = 1;

  let extraClassDeclaration = [{
    using KindTy = unsigned;

    // a !fir.boxchar<k> always wraps a !fir.char<k, ?>
    CharacterType getElementType(mlir::MLIRContext *context) const;

    CharacterType getEleTy() const;
  }];
}

#endif // FIR_DIALECT_FIR_TYPES
