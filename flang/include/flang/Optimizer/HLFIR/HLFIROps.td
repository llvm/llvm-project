//===-- HLFIROps.td - HLFIR operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the HLFIR dialect operations
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_HLFIR_OPS
#define FORTRAN_DIALECT_HLFIR_OPS

include "flang/Optimizer/HLFIR/HLFIROpBase.td"
include "flang/Optimizer/Dialect/FIRTypes.td"
include "flang/Optimizer/Dialect/FIRAttr.td"
include "flang/Optimizer/Dialect/FortranVariableInterface.td"
include "mlir/IR/BuiltinAttributes.td"

// Base class for FIR operations.
// All operations automatically get a prefix of "hlfir.".
class hlfir_Op<string mnemonic, list<Trait> traits>
  : Op<hlfir_Dialect, mnemonic, traits>;



def hlfir_DeclareOp : hlfir_Op<"declare", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<fir_FortranVariableOpInterface>]> {
  let summary = "declare a variable and produce an SSA value that can be used as a variable in HLFIR operations";

  let description = [{
    Tie the properties of a Fortran variable to an address. The properties
    include bounds, length parameters, and Fortran attributes.

    The arguments are the same as for fir.declare.

    The main difference with fir.declare is that hlfir.declare returns two
    values:
      - the first one is an SSA value that allows retrieving the variable
        address, bounds, and type parameters at any point without requiring
        access to the defining operation. This may be:
        - for scalar numerical, logical, or derived type without length
          parameters: a fir.ref<T> (e.g. fir.ref<i32>)
        - for scalar characters: a fir.boxchar<kind> or fir.ref<fir.char<kind,
          cst_len>>
        - for arrays of types without length parameters, without lower bounds,
          that are not polymorphic and with a constant shape:
          fir.ref<fir.array<cst_shapexT>>
        - for all non pointer/non allocatable entities: fir.box<T>, and
          fir.class<T> for polymorphic entities.
        - for all pointers/allocatables:
          fir.ref<fir.box<fir.ptr<T>>>/fir.ref<fir.box<fir.heap<T>>>
      - the second value has the same type as the input memref, and is the
        same. If it is a fir.box or fir.class, it may not contain accurate
        local lower bound values. It is intended to be used when generating FIR
        from HLFIR in order to avoid descriptor creation for simple entities.

    Example:

    CHARACTER(n) :: c(10:n, 20:n)

    Can be represented as:
    ```
    func.func @foo(%arg0: !fir.ref<!fir.array<?x?x!fir.char<1,?>>>, %arg1: !fir.ref<i64>) {
      %c10 = arith.constant 10 : index
      %c20 = arith.constant 20 : index
      %1 = fir.load %ag1 : fir.ref<i64>
      %2 = fir.shape_shift %c10, %1, %c20, %1 : (index, index, index, index) -> !fir.shapeshift<2>
      %3 = hfir.declare %arg0(%2) typeparams %1 {uniq_name = "c"} (fir.ref<!fir.array<?x?x!fir.char<1,?>>>, fir.shapeshift<2>, index) -> (fir.box<!fir.array<?x?x!fir.char<1,?>>>, fir.ref<!fir.array<?x?x!fir.char<1,?>>>)
      // ... uses %3#0 as "c"
    }
   ```
  }];

  let arguments = (ins
    AnyRefOrBox:$memref,
    Optional<AnyShapeOrShiftType>:$shape,
    Variadic<AnyIntegerType>:$typeparams,
    Builtin_StringAttr:$uniq_name,
    OptionalAttr<fir_FortranVariableFlagsAttr>:$fortran_attrs
  );

  let results = (outs AnyFortranVariable, AnyRefOrBoxLike);

  let assemblyFormat = [{
    $memref (`(` $shape^ `)`)? (`typeparams` $typeparams^)?
     attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$memref, "llvm::StringRef":$uniq_name,
      CArg<"mlir::Value", "{}">:$shape, CArg<"mlir::ValueRange", "{}">:$typeparams,
      CArg<"fir::FortranVariableFlagsAttr", "{}">:$fortran_attrs)>];

  let extraClassDeclaration = [{
    /// Get the variable original base (same as input). It lacks
    /// any explicit lower bounds and the extents might not be retrievable
    /// from it. This matches what is used as a "base" in FIR.
    mlir::Value getOriginalBase() {
      return getResult(1);
    }

    /// Override FortranVariableInterface default implementation
    mlir::Value getBase() {
      return getResult(0);
    }

    /// Given a FIR memory type, and information about non default lower
    /// bounds, get the related HLFIR variable type.
    static mlir::Type getHLFIRVariableType(mlir::Type type, bool hasLowerBounds);
  }];

  let hasVerifier = 1;
}

def fir_AssignOp : hlfir_Op<"assign", [MemoryEffects<[MemWrite]>]> {
  let summary = "Assign an expression or variable value to a Fortran variable";

  let description = [{
    Assign rhs to lhs following Fortran intrinsic assignments rules.
    The operation deals with inserting a temporary if the lhs and rhs
    may overlap.
  }];

  let arguments = (ins AnyFortranEntity:$rhs,
                   Arg<AnyFortranVariable, "", [MemWrite]>:$lhs);

  let assemblyFormat = [{
    $rhs `to` $lhs attr-dict `:` type(operands)
  }];
}

def hlfir_DesignateOp : hlfir_Op<"designate", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<fir_FortranVariableOpInterface>]> {
  let summary = "Designate a Fortran variable";

  let description = [{
    This operation represents a Fortran "part-ref", except that it can embed a
    substring or or complex part directly, and that vector subscripts cannot be
    used. It returns a Fortran variable that is a part of the input variable.

    The operands are as follow:
      - memref is the variable being designated.
      - component may be provided if the memref is a derived type to
        represent a reference to a component. It must be the name of a
        component of memref derived type.
      - component_shape represents the shape of the component and must be
        provided if and only if both component and indices appear.
      - indices can be provided to index arrays. The indices may be simple
        indices or triplets.
        If indices are provided and there is a component, the component must be
        an array component and the indices index the array component.
        If memref is an array, and component is provided and is an array
        component, indices must be provided and must not be triplets. This
        ensures hlfir.designate does not create arrays of arrays (which is not
        possible in Fortran).
      - substring may contain two values to represent a substring lower and
        upper bounds.
      - complex_part may be provided to represent a complex part (true
        represents the imaginary part, and false the real part).
      - shape represents the shape of the result and must be provided if the
        result is an array that is not a box address.
      - typeparams represents the length parameters of the result and must be
        provided if the result type has length parameters and is not a box
        address.
  }];

  let arguments = (ins AnyFortranVariable:$memref,
                   OptionalAttr<Builtin_StringAttr>:$component,
                   Optional<AnyShapeOrShiftType>:$component_shape,
                   Variadic<AnyIntegerType>:$indices,
                   DenseBoolArrayAttr:$is_triplet,
                   Variadic<AnyIntegerType>:$substring,
                   OptionalAttr<BoolAttr>:$complex_part,
                   Optional<AnyShapeOrShiftType>:$shape,
                   Variadic<AnyIntegerType>:$typeparams,
                   OptionalAttr<fir_FortranVariableFlagsAttr>:$fortran_attrs
                );

  let results = (outs AnyFortranVariable);

  let assemblyFormat = [{
    $memref (`{` $component^ `}`)? (`<` $component_shape^ `>`)?
    custom<DesignatorIndices>($indices, $is_triplet)
    (`substr` $substring^)?
    custom<DesignatorComplexPart>($complex_part)
    (`shape` $shape^)? (`typeparams` $typeparams^)?
    attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    using Triplet = std::tuple<mlir::Value, mlir::Value, mlir::Value>;
    using Subscript = std::variant<mlir::Value, Triplet>;
    using Subscripts = llvm::SmallVector<Subscript, 8>;
  }];

  let builders = [
    OpBuilder<(ins "mlir::Type":$result_type, "mlir::Value":$memref,
      "llvm::StringRef":$component, "mlir::Value":$component_shape,
      "llvm::ArrayRef<std::variant<mlir::Value, std::tuple<mlir::Value, mlir::Value, mlir::Value>>>":$subscripts,
      CArg<"mlir::ValueRange", "{}">:$substring,
      CArg<"std::optional<bool>", "{}">:$complex_part,
      CArg<"mlir::Value", "{}">:$shape, CArg<"mlir::ValueRange", "{}">:$typeparams,
      CArg<"fir::FortranVariableFlagsAttr", "{}">:$fortran_attrs)>,

    OpBuilder<(ins "mlir::Type":$result_type, "mlir::Value":$memref,
      "mlir::ValueRange":$indices,
      CArg<"mlir::ValueRange", "{}">:$typeparams,
      CArg<"fir::FortranVariableFlagsAttr", "{}">:$fortran_attrs)>
    ];

  let hasVerifier = 1;
}

def hlfir_ConcatOp : hlfir_Op<"concat", []> {
  let summary = "concatenate characters";
  let description = [{
    Concatenate two or more character strings of a same character kind.
  }];

  let arguments = (ins Variadic<AnyScalarCharacterEntity>:$strings,
                   AnyIntegerType:$length);

  let results = (outs AnyScalarCharacterExpr);

  let assemblyFormat = [{
    $strings `len` $length
     attr-dict `:` functional-type(operands, results)
  }];

  let builders = [OpBuilder<(ins "mlir::ValueRange":$strings,"mlir::Value":$len)>];

  let hasVerifier = 1;
}

def hlfir_SetLengthOp : hlfir_Op<"set_length", []> {
  let summary = "change the length of a character entity";
  let description = [{
    Change the length of character entity. This trims or pads the
    character argument according to the new length.
  }];

  let arguments = (ins AnyScalarCharacterEntity:$string,
                   AnyIntegerType:$length);

  let results = (outs AnyScalarCharacterExpr);

  let assemblyFormat = [{
    $string `len` $length
     attr-dict `:` functional-type(operands, results)
  }];

  let builders = [OpBuilder<(ins "mlir::Value":$string,"mlir::Value":$len)>];
}

def hlfir_AssociateOp : hlfir_Op<"associate", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<fir_FortranVariableOpInterface>]> {
  let summary = "Create a variable from an expression value";
  let description = [{
    Create a variable from an expression value.
    For expressions, this operation is an incentive to re-use the expression
    storage, if any, after the bufferization pass when possible (if the
    expression is not used afterwards).
  }];

  let arguments = (ins
    AnyFortranValue:$source,
    Optional<AnyShapeOrShiftType>:$shape,
    Variadic<AnyIntegerType>:$typeparams,
    Builtin_StringAttr:$uniq_name,
    OptionalAttr<fir_FortranVariableFlagsAttr>:$fortran_attrs
  );

  let results = (outs AnyFortranVariable, AnyRefOrBoxLike, I1);

  let assemblyFormat = [{
    $source (`(` $shape^ `)`)? (`typeparams` $typeparams^)?
     attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$source, "llvm::StringRef":$uniq_name,
      CArg<"mlir::Value", "{}">:$shape, CArg<"mlir::ValueRange", "{}">:$typeparams,
      CArg<"fir::FortranVariableFlagsAttr", "{}">:$fortran_attrs)>];

  let extraClassDeclaration = [{
    /// Override FortranVariableInterface default implementation
    mlir::Value getBase() {
      return getResult(0);
    }

    /// Get the variable FIR base (same as input). It lacks
    /// any explicit lower bounds and the extents might not be retrievable
    /// from it. This matches what is used as a "base" in FIR. All non
    /// polymorphic expressions FIR base is a simple raw address (they are
    /// contiguous in memory).
    mlir::Value getFirBase() {
      return getResult(1);
    }

    /// Return the result value that indicates if the variable storage
    /// was allocated on the heap. At the HLFIR level, this may not be
    /// known yet, and lowering will need to conditionally free the storage.
    mlir::Value getMustFreeStrorageFlag() {
      return getResult(2);
    }
  }];
}

def hlfir_EndAssociateOp : hlfir_Op<"end_associate", []> {
  let summary = "Mark the end of life of a variable associated to an expression";

  let description = [{
    Mark the end of life of a variable associated to an expression.
  }];

  let arguments = (ins AnyRefOrBoxLike:$var,
                   I1:$must_free);

  let assemblyFormat = [{
    $var `,` $must_free attr-dict `:` type(operands)
  }];

  let builders = [OpBuilder<(ins "hlfir::AssociateOp":$associate)>];
}

def hlfir_AsExprOp : hlfir_Op<"as_expr", []> {
  let summary = "Take the value of an array, character or derived variable";

  let description = [{
    Take the value of an array, character or derived variable.
    In general, this operation will lead to a copy of the variable
    in the bufferization pass if it was not transformed.

    However, if it is known that the variable storage will not be used anymore
    afterwards, the variable storage ownership can be passed to the hlfir.expr
    by providing the $must_free argument that is a boolean that indicates if
    the storage must be freed (when it was allocated on the heap).
    This allows Fortran lowering to build some expression value in memory when
    there is no adequate hlfir operation, and to promote the result to an
    hlfir.expr value without paying the price of introducing a copy.
  }];

  let arguments = (ins AnyFortranVariable:$var,
                       Optional<I1>:$must_free);
  let results = (outs hlfir_ExprType);

  let extraClassDeclaration = [{
      // Is this a "move" ?
      bool isMove() { return getMustFree() != mlir::Value{}; }
  }];

  let assemblyFormat = [{
    $var (`move` $must_free^)? attr-dict `:` functional-type(operands, results)
  }];


  let builders = [OpBuilder<(ins "mlir::Value":$var, CArg<"mlir::Value", "{}">:$must_free)>];
}

def hlfir_NoReassocOp : hlfir_Op<"no_reassoc", [NoMemoryEffect, SameOperandsAndResultType]> {
  let summary = "synthetic op to prevent reassociation";

  let description = [{
    Same as fir.reassoc, except it accepts hlfir.expr arguments.
  }];

  let arguments = (ins AnyFortranEntity:$val);
  let results = (outs AnyFortranEntity);

  let assemblyFormat = "$val attr-dict `:` type($val)";
}

def hlfir_ElementalOp : hlfir_Op<"elemental", []> {
  let summary = "elemental expression";
  let description = [{
    Represent an elemental expression as a function of the indices.
    This operation contain a region whose block arguments are one
    based indices iterating over the elemental expression shape.
    Given these indices, the element value for the given iteration
    can be computed in the region and yielded with the hlfir.yield_element
    operation.

    The shape and typeparams operands represent the extents and type
    parameters of the resulting array value.


    Example: Y + X,  with Integer :: X(10, 20), Y(10,20)
    ```
      %0 = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
      %5 = hlfir.elemental %0 : (!fir.shape<2>) -> !hlfir.expr<10x20xi32> {
      ^bb0(%i: index, %j: index):
        %6 = hlfir.designate %x (%i, %j)  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
        %7 = hlfir.designate %y (%i, %j)  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
        %8 = fir.load %6 : !fir.ref<i32>
        %9 = fir.load %7 : !fir.ref<i32>
        %10 = arith.addi %8, %9 : i32
        hlfir.yield_element %10 : i32
      }
    ```
  }];

  let arguments = (ins
    AnyShapeType:$shape,
    Variadic<AnyIntegerType>:$typeparams
  );

  let results = (outs hlfir_ExprType);
  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    $shape (`typeparams` $typeparams^)?
    attr-dict `:` functional-type(operands, results)
    $region
    }];

  let extraClassDeclaration = [{
      mlir::Block *getBody() { return &getRegion().front(); }

      // Get the indices iterating over the shape.
      mlir::Block::BlockArgListType getIndices() {
       return getBody()->getArguments();
      }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Type":$result_type, "mlir::Value":$shape,
      CArg<"mlir::ValueRange", "{}">:$typeparams)>];

}

def hlfir_YieldElementOp : hlfir_Op<"yield_element", [Terminator, HasParent<"ElementalOp">]> {
  let summary = "Yield the elemental value in an ElementalOp";
  let description = [{
    Yield the element value of the current elemental expression iteration
    in an hlfir.elemental region. See hlfir.elemental description for an
    example.
  }];

  let arguments = (ins AnyType:$element_value);

  let assemblyFormat = "$element_value attr-dict `:` type($element_value)";
}

def hlfir_ApplyOp : hlfir_Op<"apply", [NoMemoryEffect, AttrSizedOperandSegments]> {
  let summary = "get the element value of an expression";
  let description = [{
    Given an hlfir.expr array value, hlfir.apply allow retrieving
    the value for an element given one based indices.
    When hlfir.apply is used on an hlfir.elemental, and if the hlfir.elemental
    operation evaluation can be moved to the location of the hlfir.apply, it is
    as if the hlfir.elemental body was evaluated given the hlfir.apply indices.
  }];

  let arguments = (ins hlfir_ExprType:$expr,
                   Variadic<Index>:$indices,
                   Variadic<AnyIntegerType>:$typeparams
                  );
  let results = (outs AnyFortranValue:$element_value);

  let assemblyFormat = [{
    $expr `,` $indices (`typeparams` $typeparams^)?
    attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$expr, "mlir::ValueRange":$indices,
      "mlir::ValueRange":$typeparams)>
  ];
}

def hlfir_NullOp : hlfir_Op<"null", [NoMemoryEffect, fir_FortranVariableOpInterface]> {
  let summary = "create a NULL() address";

  let description = [{
    Create a NULL() address.
    So far is not intended to represent NULL(MOLD).
  }];

  let results = (outs AnyFortranVariable);
  let builders = [OpBuilder<(ins)>];

  let assemblyFormat = "type(results) attr-dict";
  let extraClassDeclaration = [{
    // Implement FortranVariableInterface interface.
    std::optional<fir::FortranVariableFlagsEnum> getFortranAttrs() const {
      return std::nullopt;
    }
    mlir::Value getShape() const {return mlir::Value{};}
    mlir::OperandRange getExplicitTypeParams() const {
      // Return an empty range.
      return {(*this)->getOperands().begin(), (*this)->getOperands().begin()};
    }
  }];
}

def hlfir_DestroyOp : hlfir_Op<"destroy", []> {
  let summary = "Mark the last use of an hlfir.expr";
  let description = [{
    Mark the last use of an hlfir.expr. This will be the point at which the
    buffer of an hlfir.expr, if any, will be deallocated if it was heap
    allocated.
    It is not required to create an hlfir.destroy operation for and hlfir.expr
    created inside an hlfir.elemental an returned in the hlfir.yield_element.
    The last use of such expression is implicit and an hlfir.destroy could
    not be emitted after the hlfir.yield_element since it is a terminator.

    Note that hlfir.destroy are currently generated by Fortran lowering that
    has a good view of the expression use contexts, but this will need to be
    revisited if any motion of hlfir.expr is done (like CSE) since
    transformations should not introduce any hlfir.expr usages after an
    hlfir.destroy.
    The future will probably be to identify the last use points automatically
    in bufferization instead.
  }];

  let arguments = (ins hlfir_ExprType:$expr);

  let assemblyFormat = "$expr attr-dict `:` qualified(type($expr))";
}

#endif // FORTRAN_DIALECT_HLFIR_OPS
