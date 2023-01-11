//===-- HLFIROpBase.td - HLFIR dialect base definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the HLFIR dialect and core hlfir.expr type
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_HLFIR_OP_BASE
#define FORTRAN_DIALECT_HLFIR_OP_BASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "flang/Optimizer/Dialect/FIRTypes.td"

def hlfir_Dialect : Dialect {
  let name = "hlfir";

  let summary = "High Level Fortran IR.";

  let description = [{
    This dialect models Fortran expressions and assignments without requiring
    the allocation and manipulation of temporary storage.
    It allows running high level optimization passes and is rather
    straightforward to generate from Fortran expressions and assignments.

    It is not a complete implementation of Fortran, for constructs and lower
    level operations, FIR should be used directly.

    A bufferization pass transforms hlfir.expr values into FIR temporary in
    memory, and its translation pass to FIR translates high level operations
    into sequence of lower level FIR operations operating on memory.
  }];

  let useDefaultTypePrinterParser = 1;
  let cppNamespace = "hlfir";
  let useFoldAPI = kEmitFoldAdaptorFolder;
}


def hlfir_ExprType : TypeDef<hlfir_Dialect, "Expr"> {
  let mnemonic = "expr";
  let summary = "The type of an array, character, or derived type Fortran expression";

  let description = [{
    Abstract value type for Fortran arrays, characters and derived types.
    The rank cannot be assumed, and empty shape means that the expression is a scalar.
    When the element type is a derived type, the polymorphic flag may be set to true
    to indicate that the expression dynamic type can differ from its static type.
  }];


  let parameters = (ins
    ArrayRefParameter<"int64_t", "expression shape">:$shape,
    "mlir::Type":$elementType,
    "bool":$polymorphic
  );

  let extraClassDeclaration = [{
    using Shape = llvm::SmallVector<int64_t>;
    mlir::Type getEleTy() const {return getElementType();}
    bool isScalar() const { return getShape().empty(); }
    bool isArray() const { return !isScalar(); }
    bool isPolymorphic() const { return getPolymorphic(); }
    unsigned getRank() const {return getShape().size();}
    mlir::Type getElementExprType() const {
      mlir::Type eleTy = getElementType();
      if (fir::isa_trivial(eleTy))
        return eleTy;
      return hlfir::ExprType::get(eleTy.getContext(), Shape{}, eleTy,
                isPolymorphic());
    }
    static constexpr int64_t getUnknownExtent() {
      return mlir::ShapedType::kDynamic;
    }
  }];

  let hasCustomAssemblyFormat = 1;

}

def IsFortranVariablePred
        : CPred<"::hlfir::isFortranVariableType($_self)">;
def AnyFortranVariable : Type<IsFortranVariablePred, "any HLFIR variable type">;


def AnyFortranValue : TypeConstraint<Or<[AnyLogicalLike.predicate,
    AnyIntegerLike.predicate, AnyRealLike.predicate,
    fir_ComplexType.predicate, AnyComplex.predicate,
    hlfir_ExprType.predicate]>, "any Fortran value type">;


def AnyFortranEntity : TypeConstraint<Or<[AnyFortranVariable.predicate,
    AnyFortranValue.predicate]>, "any Fortran value or variable type">;

def IsFortranScalarCharacterPred
        : CPred<"::hlfir::isFortranScalarCharacterType($_self)">;
def AnyScalarCharacterEntity : Type<IsFortranScalarCharacterPred,
    "any character scalar type">;

def IsFortranScalarCharacterExprPred
        : CPred<"::hlfir::isFortranScalarCharacterExprType($_self)">;
def AnyScalarCharacterExpr : Type<IsFortranScalarCharacterExprPred,
    "any character scalar expression type">;

#endif // FORTRAN_DIALECT_HLFIR_OP_BASE
