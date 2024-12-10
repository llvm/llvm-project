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
include "mlir/IR/EnumAttr.td"
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
  let usePropertiesForAttributes = 1;
  let cppNamespace = "hlfir";
  let dependentDialects = ["fir::FIROpsDialect"];
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
    AnyFirComplexLike.predicate,
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

def isFortranArrayCharacterExprTypePred
        : CPred<"::hlfir::isFortranArrayCharacterExprType($_self)">;
def AnyArrayCharacterExpr : Type<isFortranArrayCharacterExprTypePred,
    "any character array expression type">;

def AnyScalarOrArrayCharacterExpr :
    Type<Or<[AnyScalarCharacterExpr.predicate,
             AnyArrayCharacterExpr.predicate]>,
    "any character scalar or array expression type">;

def IsFortranNumericalArrayObjectPred
        : CPred<"::hlfir::isFortranNumericalArrayObject($_self)">;
def AnyFortranNumericalArrayObject : Type<IsFortranNumericalArrayObjectPred,
    "any array-like object containing a numerical type">;

def IsFortranNumericalOrLogicalArrayObjectPred
        : CPred<"::hlfir::isFortranNumericalOrLogicalArrayObject($_self)">;
def AnyFortranNumericalOrLogicalArrayObject : Type<IsFortranNumericalOrLogicalArrayObjectPred,
    "any array-like object containing a numerical or logical type">;

def IsFortranArrayObjectPred
        : CPred<"::hlfir::isFortranArrayObject($_self)">;
def AnyFortranArrayObject : Type<IsFortranArrayObjectPred,
    "any array-like object">;

def IsPassByRefOrIntegerTypePred
        : CPred<"::hlfir::isPassByRefOrIntegerType($_self)">;
def AnyPassByRefOrIntegerType : Type<IsPassByRefOrIntegerTypePred,
    "an integer type either by value or by reference">;

def IsMaskArgumentPred
        : CPred<"::hlfir::isMaskArgument($_self)">;
def AnyFortranLogicalOrI1ArrayObject : Type<IsMaskArgumentPred,
    "A scalar i1 or logical or an array-like object containing logicals">;

def IsFortranLogicalArrayPred
        : CPred<"::hlfir::isFortranLogicalArrayObject($_self)">;
def AnyFortranLogicalArrayObject : Type<IsFortranLogicalArrayPred,
    "any array-like object containing logicals">;

def IsPolymorphicObjectPred
        : CPred<"::hlfir::isPolymorphicObject($_self)">;
def AnyPolymorphicObject : Type<IsPolymorphicObjectPred,
    "any polymorphic object">;

def IsFortranIntegerScalarOrArrayPred
    : CPred<"::hlfir::isFortranIntegerScalarOrArrayObject($_self)">;
def AnyFortranIntegerScalarOrArrayObject
    : Type<IsFortranIntegerScalarOrArrayPred,
           "A scalar or array object containing integers">;

def hlfir_CharExtremumPredicateAttr : I32EnumAttr<
    "CharExtremumPredicate", "",
    [
      I32EnumAttrCase<"min", 0>,
      I32EnumAttrCase<"max", 1>
    ]> {
  let cppNamespace = "hlfir";
}
#endif // FORTRAN_DIALECT_HLFIR_OP_BASE
