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
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

// Base class for FIR operations.
// All operations automatically get a prefix of "hlfir.".
class hlfir_Op<string mnemonic, list<Trait> traits>
  : Op<hlfir_Dialect, mnemonic, traits>;


// DeclareOp can be lowered to EmboxOp, EmboxCharOp, ReboxOp, etc and so could
// generate code. All of the operations it can generate are modelled with
// NoMemoryEffect. However, if hlfir.declare is given NoMemoryEffect, it can be
// removed by dead code elimination if the value result is unused. Information
// from the declare operation can be used to generate debug information so we
// don't want to remove it as dead code
def hlfir_DeclareOp : hlfir_Op<"declare", [AttrSizedOperandSegments,
    MemoryEffects<[MemWrite<DebuggingResource>]>,
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
    OptionalAttr<fir_FortranVariableFlagsAttr>:$fortran_attrs,
    OptionalAttr<fir_CUDADataAttributeAttr>:$cuda_attr
  );

  let results = (outs AnyFortranVariable, AnyRefOrBoxLike);

  let assemblyFormat = [{
    $memref (`(` $shape^ `)`)? (`typeparams` $typeparams^)?
     attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$memref, "llvm::StringRef":$uniq_name,
      CArg<"mlir::Value", "{}">:$shape, CArg<"mlir::ValueRange", "{}">:$typeparams,
      CArg<"fir::FortranVariableFlagsAttr", "{}">:$fortran_attrs,
      CArg<"fir::CUDADataAttributeAttr", "{}">:$cuda_attr)>];

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
    The optional "realloc" flag allows indicating that this assignment
    has the Fortran 95 semantics for assignments to a whole allocatable.
    In such case, the left hand side must be an allocatable that may be
    unallocated or allocated with a different type and shape than the right
    hand side. It will be allocated or re-allocated as needed during the
    assignment.
    When "realloc" is set and this is a character assignment, the optional
    flag "keep_lhs_length_if_realloc" indicates that the character
    left hand side should retain its length after the assignment. If the
    right hand side has a different length, truncation and padding will
    occur. This covers the case of explicit and assumed length character
    allocatables.
    Otherwise, the left hand side will be allocated or reallocated to match the
    right hand side length if they differ. This covers the case of deferred
    length character allocatables.
    The optional "temporary_lhs" flag indicates that the LHS is a compiler
    generated temporary. In this case the temporary is initialized if needed
    (e.g. the LHS is of derived type with allocatable/pointer components),
    and the assignment is done without LHS (or its subobjects) finalization
    and with automatic allocation.
    If "temporary_lhs" and "keep_lhs_length_if_realloc" are both set,
    this assign operation denotes special case of character allocatable
    LHS with explicit length. The LHS that must preserve its length
    during the assignment regardless of the the RHS's length or/and
    allocation status. This assign operation will be lowered into a call
    to AssignExplicitLengthCharacter().
  }];

  let arguments = (ins AnyFortranEntity:$rhs,
                   Arg<AnyFortranVariable, "", [MemWrite]>:$lhs,
                   UnitAttr:$realloc,
                   UnitAttr:$keep_lhs_length_if_realloc,
                   UnitAttr:$temporary_lhs);

  let assemblyFormat = [{
    $rhs `to` $lhs (`realloc` $realloc^)?
    (`keep_lhs_len` $keep_lhs_length_if_realloc^)?
    (`temporary_lhs` $temporary_lhs^)?
    attr-dict `:` type(operands)
  }];

  let extraClassDeclaration = [{
    /// Does this assignment have the Fortran 95 semantics of assignments
    /// to a whole allocatable?
    bool isAllocatableAssignment() {
      return getRealloc();
    }
    /// Is the assignment left hand side a whole allocatable character
    /// that should retain its length after the assignment?
    bool mustKeepLhsLengthInAllocatableAssignment() {
      return getKeepLhsLengthIfRealloc();
    }
    /// Is the assignment's left hand side a compiler generated temporary?
    bool isTemporaryLHS() {
      return getTemporaryLhs();
    }
  }];

  let hasVerifier = 1;
}

def hlfir_DesignateOp : hlfir_Op<"designate", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<fir_FortranVariableOpInterface>, NoMemoryEffect]> {
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

def hlfir_ParentComponentOp : hlfir_Op<"parent_comp", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<fir_FortranVariableOpInterface>]> {
  let summary = "Designate the parent component of a variable";

  let description = [{
    This operation represents a Fortran component reference where the
    component name is a parent type of the variable's derived type.
    These component references cannot be represented with an hlfir.designate
    because the parent type names are not embedded in fir.type<> types
    as opposed to the actual component names.

    The operands are as follow:
      - memref is a derived type variable whose parent component is being
        designated.
      - shape is the shape of memref and the result and must be provided if
        memref is an array. Parent component reference lower bounds are ones,
        so the provided shape must be a fir.shape.
      - typeparams are the type parameters of the parent component type if any.
        It is a subset of memref type parameters.
    The parent component type and name is reflected in the result type.
  }];

  let arguments = (ins AnyFortranVariable:$memref,
                   Optional<AnyShapeType>:$shape,
                   Variadic<AnyIntegerType>:$typeparams);

  let extraClassDeclaration = [{
    // Implement FortranVariableInterface interface. Parent components have
    // no attributes (pointer, allocatable or contiguous can only be added
    // to regular components).
    std::optional<fir::FortranVariableFlagsEnum> getFortranAttrs() const {
      return std::nullopt;
    }
  }];

  let results = (outs AnyFortranVariable);

  let assemblyFormat = [{
    $memref (`shape` $shape^)? (`typeparams` $typeparams^)?
    attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def hlfir_ConcatOp : hlfir_Op<"concat",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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

def hlfir_AllOp : hlfir_Op<"all", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ALL transformational intrinsic";
  let description = [{
    Takes a logical array MASK as argument, optionally along a particular dimension,
    and returns true if all elements of MASK are true.
  }];

  let arguments = (ins
    AnyFortranLogicalArrayObject:$mask,
    Optional<AnyIntegerType>:$dim
  );

  let results = (outs AnyFortranValue);

  let assemblyFormat = [{
    $mask  (`dim` $dim^)?  attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def hlfir_AnyOp : hlfir_Op<"any", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ANY transformational intrinsic";
  let description = [{
    Takes a logical array MASK as argument, optionally along a particular dimension,
    and returns true if any element of MASK is true.
  }];

  let arguments = (ins
    AnyFortranLogicalArrayObject:$mask,
    Optional<AnyIntegerType>:$dim
  );

  let results = (outs AnyFortranValue);

  let assemblyFormat = [{
    $mask  (`dim` $dim^)?  attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def hlfir_CountOp : hlfir_Op<"count", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "COUNT transformational intrinsic";
  let description = [{
    Takes a logical and counts the number of true values.
  }];

  let arguments = (ins
    AnyFortranLogicalArrayObject:$mask,
    Optional<AnyIntegerType>:$dim
  );

  let results = (outs AnyFortranValue);

  let assemblyFormat = [{
    $mask  (`dim` $dim^)?  attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def hlfir_MaxvalOp : hlfir_Op<"maxval", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<ArithFastMathInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "MAXVAL transformational intrinsic";
  let description = [{
    Maximum value(s) of an array.
    If DIM is absent, the result is a scalar.
    If DIM is present, the result is an array of rank n-1, where n is the rank of ARRAY.
  }];

  let arguments = (ins
    AnyFortranArrayObject:$array,
    Optional<AnyIntegerType>:$dim,
    Optional<AnyFortranLogicalOrI1ArrayObject>:$mask,
    DefaultValuedAttr<Arith_FastMathAttr,
                      "::mlir::arith::FastMathFlags::none">:$fastmath
  );

  let results = (outs AnyFortranValue);

  let assemblyFormat = [{
    $array (`dim` $dim^)? (`mask` $mask^)? attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def hlfir_MinvalOp : hlfir_Op<"minval", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<ArithFastMathInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "MINVAL transformational intrinsic";
  let description = [{
    Minimum value(s) of an array.
    If DIM is absent, the result is a scalar.
    If DIM is present, the result is an array of rank n-1, where n is the rank of ARRAY.
  }];

  let arguments = (ins
    AnyFortranArrayObject:$array,
    Optional<AnyIntegerType>:$dim,
    Optional<AnyFortranLogicalOrI1ArrayObject>:$mask,
    DefaultValuedAttr<Arith_FastMathAttr,
                      "::mlir::arith::FastMathFlags::none">:$fastmath
  );

  let results = (outs AnyFortranValue);

  let assemblyFormat = [{
    $array (`dim` $dim^)? (`mask` $mask^)? attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def hlfir_MinlocOp : hlfir_Op<"minloc", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<ArithFastMathInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "MINLOC transformational intrinsic";
  let description = [{
    Minlocs of an array.
  }];

  let arguments = (ins
    AnyFortranArrayObject:$array,
    Optional<AnyIntegerType>:$dim,
    Optional<AnyFortranLogicalOrI1ArrayObject>:$mask,
    Optional<Type<AnyLogicalLike.predicate>>:$back,
    DefaultValuedAttr<Arith_FastMathAttr,
                      "::mlir::arith::FastMathFlags::none">:$fastmath
  );

  let results = (outs AnyFortranValue);

  let assemblyFormat = [{
    $array (`dim` $dim^)? (`mask` $mask^)? (`back` $back^)?  attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def hlfir_MaxlocOp : hlfir_Op<"maxloc", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<ArithFastMathInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "MAXLOC transformational intrinsic";
  let description = [{
    Maxlocs of an array.
  }];

  let arguments = (ins
    AnyFortranArrayObject:$array,
    Optional<AnyIntegerType>:$dim,
    Optional<AnyFortranLogicalOrI1ArrayObject>:$mask,
    Optional<Type<AnyLogicalLike.predicate>>:$back,
    DefaultValuedAttr<Arith_FastMathAttr,
                      "::mlir::arith::FastMathFlags::none">:$fastmath
  );

  let results = (outs AnyFortranValue);

  let assemblyFormat = [{
    $array (`dim` $dim^)? (`mask` $mask^)? (`back` $back^)?  attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def hlfir_ProductOp : hlfir_Op<"product", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<ArithFastMathInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "PRODUCT transformational intrinsic";
  let description = [{
    Multiplies the elements of an array, optionally along a particular dimension,
    optionally if a mask is true.
  }];

  let arguments = (ins
    AnyFortranNumericalArrayObject:$array,
    Optional<AnyIntegerType>:$dim,
    Optional<AnyFortranLogicalOrI1ArrayObject>:$mask,
    DefaultValuedAttr<Arith_FastMathAttr,
                      "::mlir::arith::FastMathFlags::none">:$fastmath
  );

  let results = (outs AnyFortranValue);

  let assemblyFormat = [{
    $array (`dim` $dim^)? (`mask` $mask^)? attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def hlfir_SetLengthOp : hlfir_Op<"set_length",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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

def hlfir_GetLengthOp : hlfir_Op<"get_length", [Pure]> {
  let summary = "get the length of a character entity";
  let description = [{
    Get the length of character entity represented as hlfir.expr.
  }];

  let arguments = (ins AnyScalarOrArrayCharacterExpr:$expr);
  let results = (outs Index);

  let assemblyFormat = [{
    $expr attr-dict `:` functional-type(operands, results)
  }];

  // If character length is know via the type, then the operation
  // may be immediately canonicalized into arith::ConstantOp.
  let hasCanonicalizeMethod = 1;
}

def hlfir_SumOp : hlfir_Op<"sum", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<ArithFastMathInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "SUM transformational intrinsic";
  let description = [{
    Sums the elements of an array, optionally along a particular dimension,
    optionally if a mask is true.
  }];

  let arguments = (ins
    AnyFortranNumericalArrayObject:$array,
    Optional<AnyIntegerType>:$dim,
    Optional<AnyFortranLogicalOrI1ArrayObject>:$mask,
    DefaultValuedAttr<Arith_FastMathAttr,
                      "::mlir::arith::FastMathFlags::none">:$fastmath
  );

  let results = (outs AnyFortranValue);

  let assemblyFormat = [{
    $array (`dim` $dim^)? (`mask` $mask^)? attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def hlfir_DotProductOp : hlfir_Op<"dot_product",
    [DeclareOpInterfaceMethods<ArithFastMathInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "DOT_PRODUCT transformational intrinsic";
  let description = [{
    Dot product of two vectors
  }];

  let arguments = (ins
    AnyFortranNumericalOrLogicalArrayObject:$lhs,
    AnyFortranNumericalOrLogicalArrayObject:$rhs,
    DefaultValuedAttr<Arith_FastMathAttr,
                      "::mlir::arith::FastMathFlags::none">:$fastmath
  );

  let results = (outs AnyFortranValue);

  let assemblyFormat = [{
    $lhs $rhs attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def hlfir_MatmulOp : hlfir_Op<"matmul",
    [DeclareOpInterfaceMethods<ArithFastMathInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "MATMUL transformational intrinsic";
  let description = [{
    Matrix multiplication
  }];

  let arguments = (ins
    AnyFortranNumericalOrLogicalArrayObject:$lhs,
    AnyFortranNumericalOrLogicalArrayObject:$rhs,
    DefaultValuedAttr<Arith_FastMathAttr,
                      "::mlir::arith::FastMathFlags::none">:$fastmath
  );

  let results = (outs hlfir_ExprType);

  let assemblyFormat = [{
    $lhs $rhs attr-dict `:` functional-type(operands, results)
  }];

  // MATMUL(TRANSPOSE(...), ...) => hlfir.matmul_transpose
  let hasCanonicalizeMethod = 1;

  let hasVerifier = 1;
}

def hlfir_TransposeOp : hlfir_Op<"transpose",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "TRANSPOSE transformational intrinsic";
  let description = [{
    Transpose a rank 2 array
  }];

  let arguments = (ins AnyFortranArrayObject:$array);

  let results = (outs hlfir_ExprType);

  let assemblyFormat = [{
    $array attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def hlfir_MatmulTransposeOp : hlfir_Op<"matmul_transpose",
    [DeclareOpInterfaceMethods<ArithFastMathInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Optimized MATMUL(TRANSPOSE(...), ...)";
  let description = [{
    Matrix multiplication where the left hand side is transposed
  }];

  let arguments = (ins
    AnyFortranNumericalOrLogicalArrayObject:$lhs,
    AnyFortranNumericalOrLogicalArrayObject:$rhs,
    DefaultValuedAttr<Arith_FastMathAttr,
                      "::mlir::arith::FastMathFlags::none">:$fastmath
  );

  let results = (outs hlfir_ExprType);

  let assemblyFormat = [{
    $lhs $rhs attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

// An allocation effect is needed because the value produced by the associate
// is "deallocated" by hlfir.end_associate (the end_associate must not be
// removed, and there must be only one hlfir.end_associate).
def hlfir_AssociateOp : hlfir_Op<"associate", [AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<fir_FortranVariableOpInterface>,
    MemoryEffects<[MemAlloc]>]> {
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
    OptionalAttr<Builtin_StringAttr>:$uniq_name,
    OptionalAttr<fir_FortranVariableFlagsAttr>:$fortran_attrs
  );

  let results = (outs AnyFortranVariable, AnyRefOrBoxLike, I1);

  let assemblyFormat = [{
    $source (`(` $shape^ `)`)? (`typeparams` $typeparams^)?
     attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$source, CArg<"llvm::StringRef", "{}">:$uniq_name,
      CArg<"mlir::Value", "{}">:$shape, CArg<"mlir::ValueRange", "{}">:$typeparams,
      CArg<"fir::FortranVariableFlagsAttr", "{}">:$fortran_attrs)>,
    OpBuilder<(ins "mlir::Value":$memref, CArg<"mlir::Value", "{}">:$shape,
      CArg<"mlir::ValueRange", "{}">:$typeparams, 
      CArg<"fir::FortranVariableFlagsAttr", "{}">:$fortran_attrs,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes)>];

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

def hlfir_EndAssociateOp : hlfir_Op<"end_associate", [MemoryEffects<[MemFree]>]> {
  let summary = "Mark the end of life of a variable associated to an expression";

  let description = [{
    Mark the end of life of a variable associated to an expression.
    If the expression has a derived type that may contain allocatable
    components, the variable operand must be a Fortran entity.
  }];

  let arguments = (ins AnyRefOrBoxLike:$var,
                   I1:$must_free);

  let assemblyFormat = [{
    $var `,` $must_free attr-dict `:` type(operands)
  }];

  let builders = [OpBuilder<(ins "hlfir::AssociateOp":$associate)>];
  let hasVerifier = 1;
}

def hlfir_AsExprOp : hlfir_Op<"as_expr",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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

def hlfir_ElementalOpInterface : OpInterface<"ElementalOpInterface"> {
  let description = [{
    Interface for the operation holding a region with elemental computation.
    It is used as a common interface bewteen hlfir.elemental and hlfir.elemental_addr.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the one based elemental indices.",
      /*retTy=*/"mlir::Block::BlockArgListType",
      /*methodName=*/"getIndices",
      /*args=*/(ins),
      /*methodBody=*/[{}]
    >,
    InterfaceMethod<
      /*desc=*/"Return the element entity being computed",
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getElementEntity",
      /*args=*/(ins),
      /*methodBody=*/[{}]
    >,
    InterfaceMethod<
      /*desc=*/"Get element cleanup region, if any.",
      /*retTy=*/"mlir::Region*",
      /*methodName=*/"getElementCleanup",
      /*args=*/(ins),
      /*methodBody=*/[{}]
    >,
    InterfaceMethod<
      /*desc=*/"Get elemental region.",
      /*retTy=*/"mlir::Region&",
      /*methodName=*/"getElementalRegion",
      /*args=*/(ins),
      /*methodBody=*/[{}]
    >,
    InterfaceMethod<
      /*desc=*/"Must this elemental operation be evaluated in order?",
      /*retTy=*/"bool",
      /*methodName=*/"isOrdered",
      /*args=*/(ins),
      /*methodBody=*/[{}]
    >,
  ];

  let cppNamespace = "hlfir";
}

def hlfir_ElementalOp : hlfir_Op<"elemental",
    // The ElementalOp, in general, causes an allocation of a temporary,
    // so to guarantee proper behavior of MLIR optimization passes
    // we explicitly set MemAlloc effect for it. On top of this,
    // the recursive memory effects also apply.
    [RecursiveMemoryEffects, MemoryEffects<[MemAlloc]>,
     hlfir_ElementalOpInterface, AttrSizedOperandSegments]> {
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

    The optional mold is an entity carrying the information about
    the dynamic type of the polymorphic result. Note that the shape
    of the mold does not necessarily match the shape of the result,
    for example, the result of `merge(poly_scalar1, poly_scalar2, mask_array)`
    will have the shape of `mask_array` and the dynamic type of `poly_scalar*`.

    The unordered attribute can be set to allow out of order processing
    of the indices. This is safe only if the operations in the body
    of the elemental do not have side effects.


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
    Optional<AnyPolymorphicObject>:$mold,
    Variadic<AnyIntegerType>:$typeparams,
    OptionalAttr<UnitAttr>:$unordered
  );

  let results = (outs hlfir_ExprType);
  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    $shape (`mold` $mold^)? (`typeparams` $typeparams^)?
    (`unordered` $unordered^)?
    attr-dict `:` functional-type(operands, results)
    $region
    }];

  let extraClassDeclaration = [{
      mlir::Block *getBody() { return &getRegion().front(); }

      /// Get the indices iterating over the shape.
      mlir::Block::BlockArgListType getIndices() {
       return getBody()->getArguments();
      }
      /// ElementalOpInterface implementation.

      mlir::Region& getElementalRegion() { return getRegion(); }
      mlir::Value getElementEntity();
      mlir::Region* getElementCleanup() { return nullptr; }

      /// Must this elemental be evaluated in order?
      bool isOrdered() { return !getUnordered(); }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Type":$result_type, "mlir::Value":$shape,
      CArg<"mlir::Value", "{}">:$mold,
      CArg<"mlir::ValueRange", "{}">:$typeparams,
      CArg<"bool", "false">:$isUnordered)>
  ];

  let hasVerifier = 1;
}

def hlfir_YieldElementOp : hlfir_Op<"yield_element", [Terminator, HasParent<"ElementalOp">, Pure]> {
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
    Therefore, apply operations on hlfir.elemental expressions should be located
    such that evaluating the hlfir.elemental at the position of the hlfir.apply
    operation produces the same result as evaluating the hlfir.elemental at its
    location in the instruction stream. Attention should be paid to
    hlfir.elemental memory side effects (in practice these are unlikely).
    "10.1.4 Evaluation of operations" says that expression evaluation shall not
    impact/be impacted by other expression evaluation in the statement.
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

def hlfir_DestroyOp : hlfir_Op<"destroy", [MemoryEffects<[MemFree]>]> {
  let summary = "Mark the last use of an hlfir.expr";
  let description = [{
    Mark the last use of an hlfir.expr. This will be the point at which the
    buffer of an hlfir.expr, if any, will be deallocated if it was heap
    allocated.
    If "finalize" attribute is set, the hlfir.expr value will be finalized
    before the deallocation. Note that this implies that the hlfir.expr
    is placed into a memory buffer, so that the library runtime
    can be called on it. The element type of the hlfir.expr must be
    derived type in this case.
    It is not required to create an hlfir.destroy operation for and hlfir.expr
    created inside an hlfir.elemental and returned in the hlfir.yield_element.
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

  let arguments = (ins
    hlfir_ExprType:$expr,
    UnitAttr:$finalize
  );

  let assemblyFormat = [{
    $expr (`finalize` $finalize^)? attr-dict `:` qualified(type($expr))
  }];

  let extraClassDeclaration = [{
    bool mustFinalizeExpr() {
      return getFinalize();
    }
  }];

  let hasVerifier = 1;
}

def hlfir_CopyInOp : hlfir_Op<"copy_in", [MemoryEffects<[MemAlloc]>]> {
  let summary = "copy a variable into a contiguous temporary if it is not contiguous";
  let description = [{
    Copy a variable into a contiguous temporary if the variable is not
    an absent optional and is not contiguous at runtime. When a copy is made this
    operation returns the temporary as first result, otherwise, it returns the
    potentially absent variable storage. The second result indicates if a copy
    was made.

    This operation is meant to be used in combination with the hlfir.copy_out
    operation that deletes the temporary if it was created and copies the data
    back if needed.
    This operation allows passing non contiguous arrays to contiguous dummy
    arguments, which is possible in Fortran procedure references.

    To deal with the optional case, an extra boolean value can be pass to the
    operation. In such cases, the copy-in will only be done if "var_is_present"
    is true and, when it is false, the original value will be returned instead.
  }];

  let arguments = (ins Arg<fir_BaseBoxType, "", [MemRead]>:$var,
                   Optional<I1>:$var_is_present);

  let results = (outs fir_BaseBoxType, I1);

  let assemblyFormat = [{
    $var (`handle_optional` $var_is_present^)?
    attr-dict `:` functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$var, "mlir::Value":$var_is_present)>
  ];

  let extraClassDeclaration = [{
    /// Get the resulting copied-in fir.box or fir.class.
    mlir::Value getCopiedIn() {
      return getResult(0);
    }

    /// Get the result indicating if a copy was made.
    mlir::Value getWasCopied() {
      return getResult(1);
    }
  }];
}

def hlfir_CopyOutOp : hlfir_Op<"copy_out", [MemoryEffects<[MemFree]>]> {
  let summary = "copy out a variable after a copy in";
  let description = [{
    If the variable was copied in a temporary in the related hlfir.copy_in,
    optionally copy back the temporary value to it (that may have been
    modified between the hlfir.copy_in and hlfir.copy_out). Then deallocate
    the temporary.
    The copy back is done if $var is provided and $was_copied is true.
    The deallocation of $temp is done if $was_copied is true.
  }];

  let arguments = (ins Arg<fir_BaseBoxType, "", [MemRead]>:$temp,
                       I1:$was_copied,
                       Arg<Optional<fir_BaseBoxType>, "", [MemWrite]>:$var);

  let assemblyFormat = [{
    $temp `,` $was_copied (`to` $var^)?
    attr-dict `:` functional-type(operands, results)
  }];
}

def hlfir_ShapeOfOp : hlfir_Op<"shape_of", [Pure]> {
  let summary = "Get the shape of a hlfir.expr";
  let description = [{
    Gets the runtime shape of a hlfir.expr. In lowering to FIR, the
    hlfir.shape_of operation will be replaced by an fir.shape.
    It is not valid to request the shape of a hlfir.expr which has no shape.
  }];

  let arguments = (ins hlfir_ExprType:$expr);

  let results = (outs fir_ShapeType);

  let hasVerifier = 1;

  // If all extents are known at compile time, the hlfir.shape_of can be
  // immediately folded into a fir.shape operation. This makes information
  // available sooner to inform bufferization decisions
  let hasCanonicalizeMethod = 1;

  let extraClassDeclaration = [{
    std::size_t getRank();
  }];

  let assemblyFormat = [{
    $expr attr-dict `:` functional-type(operands, results)
  }];

  let builders = [OpBuilder<(ins "mlir::Value":$expr)>];
}

def hlfir_GetExtentOp : hlfir_Op<"get_extent", [Pure]> {
  let summary = "Get an extent value from a fir.shape";
  let description = [{
    Gets an extent value from a fir.shape. The dimension argument uses C style
    indexing and so should be between 0 and 1 less than the rank of the shape
  }];

  let arguments = (ins fir_ShapeType:$shape,
                       IndexAttr:$dim);

  let results = (outs Index);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $shape attr-dict `:` functional-type(operands, results)
  }];

  let builders = [OpBuilder<(ins "mlir::Value":$shape, "unsigned":$dim)>];
}

def hlfir_OrderedAssignmentTreeOpInterface : OpInterface<"OrderedAssignmentTreeOpInterface"> {
  let description = [{
    Interface for the operations representing Forall and Where constructs and
    statements as an mlir::Region tree.

    These operations all have in common that they have "leaf" regions that contains
    some code that should be evaluated for "all active combinations of Forall
    index-name values" before the next OrderedAssignmentTreeOpInterface is
    evaluated.

    These operations are ordered in a tree fashion: Some operations, like
    hlfir.forall or hlfir.where, contain a list of OrderedAssignmentTreeOpInterface
    that should be evaluated after the "Leaf" regions, and before the next
    OrderedAssignmentTreeOpInterface.

    Nested OrderedAssignmentTreeOpInterface operations are affected by the
    OrderedAssignmentTreeOpInterface operations that contain them (e.g:
    hlfir.region_assign may be masked by the value of the mask region of
    an hlfir.where that contains it).

    OrderedAssignmentTreeOpInterface operations that contain nested operation
    must return a "sub-tree" region that contains the list of nested
    OrderedAssignmentTreeOpInterface operations.

    There is no constraints over what IR a leaf region may contain. There is also
    no restriction regarding how many leaf regions an
    OrderedAssignmentTreeOpInterface operation may contain.

    A "sub-tree" region, if any, must contain only OrderedAssignmentTreeOpInterface
    operations and, maybe, a fir.end terminator.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get the OrderedAssignmentTreeOpInterface leaf regions that contain evaluation code",
      /*retTy=*/"void",
      /*methodName=*/"getLeafRegions",
      /*args=*/(ins "llvm::SmallVectorImpl<mlir::Region*>&":$regions),
      /*methodBody=*/[{}]
    >,
    InterfaceMethod<
      /*desc=*/"Get the region, if any, containing the list of sub-tree OrderedAssignmentTreeOpInterface nodes",
      /*retTy=*/"mlir::Region*",
      /*methodName=*/"getSubTreeRegion",
      /*args=*/(ins),
      /*methodBody=*/[{}]
    >,
  ];

  let extraClassDeclaration = [{
    /// Interface verifier imlementation.
    mlir::LogicalResult verifyImpl();

    mlir::Block* getSubTreeBlock() {
      mlir::Region* region = getSubTreeRegion();
      return region && !region->empty()? &region->front() : nullptr;
    }
  }];

  let verify = [{
    return ::mlir::cast<::hlfir::OrderedAssignmentTreeOpInterface>($_op).verifyImpl();
  }];

  let cppNamespace = "hlfir";
}


def hlfir_RegionAssignOp : hlfir_Op<"region_assign", [hlfir_OrderedAssignmentTreeOpInterface]> {
  let summary = "represent a Fortran assignment using regions for the LHS and RHS evaluation";
  let description = [{
    This operation can represent Forall and Where assignment when inside an
    hlfir.forall or hlfir.where "ordered assignment tree". It can
    also represent user defined assignments and assignment to vector
    subscripted entities without requiring the materialization of the
    right-hand side temporary copy that may be needed to implement Fortran
    assignment semantic.

    The right-hand side and left-hand side evaluations are held in their
    own regions terminated with hlfir.yield operations (or hlfir.elemental_addr
    for a left-hand side with vector subscript).

    An optional region may be added to implement user defined assignment.
    This region provides two block arguments with the same type as the
    yielded rhs and lhs entities (in that order), or the element type if this
    is an elemental user defined assignment.

    If this optional region is not provided, intrinsic assignment is performed.

    Example: "X = Y",  where "=" is a user defined elemental assignment "foo"
    taking Y by value.
    ```
    hlfir.region_assign {
      hlfir.yield %y : !fir.box<!fir.array<?x!f32>>
    } to {
      hlfir.yield %x : !fir.box<!fir.array<?x!fir.type<t>>>
    } user_defined_assignment (%rhs_elt: !fir.ref<f32>) to (%lhs_elt: !fir.ref<!fir.type<t>>) {
      %0 = fir.load %rhs_elt : !fir.ref<f32>
      fir.call @foo(%lhs_elt, %0) : (!fir.ref<!fir.type<t>>, f32) -> ()
    }
    ```

    TODO: add optional "realloc" semantics like for hlfir.assign.
  }];

  let regions = (region  SizedRegion<1>:$rhs_region,
                         SizedRegion<1>:$lhs_region,
                         MaxSizedRegion<1>:$user_defined_assignment);

  let extraClassDeclaration = [{
    mlir::Value getUserAssignmentRhs() {
      return getUserDefinedAssignment().getArguments()[0];
    }
    mlir::Value getUserAssignmentLhs() {
      return getUserDefinedAssignment().getArguments()[1];
    }
    void getLeafRegions(llvm::SmallVectorImpl<mlir::Region*>& regions) {
      regions.push_back(&getRhsRegion());
      regions.push_back(&getLhsRegion());
      if (!getUserDefinedAssignment().empty())
        regions.push_back(&getUserDefinedAssignment());
    }
    mlir::Region* getSubTreeRegion() { return nullptr; }

  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def hlfir_YieldOp : hlfir_Op<"yield", [Terminator, ParentOneOf<["RegionAssignOp",
    "ElementalAddrOp", "ForallOp", "ForallMaskOp", "WhereOp", "ElseWhereOp"]>,
    SingleBlockImplicitTerminator<"fir::FirEndOp">, RecursivelySpeculatable,
        RecursiveMemoryEffects]> {

  let summary = "Yield a value or variable inside a forall, where or region assignment";

  let description = [{
    Terminator operation that yields an HLFIR value or variable that was computed in
    a region and hold the yielded entity cleanup, if any, into its own region.
    This allows representing any Fortran expression evaluation in its own region so
    that the evaluation can easily be scheduled/moved around in a pass.

    Example: "foo(x)" where foo returns an allocatable array.
    ```
    {
      // In some region.
      %0 = fir.call @foo(x) (!fir.ref<f32>) -> !fir.box<fir.heap<!fir.array<?xf32>>>
      hlfir.yield %0 : !fir.box<!fir.heap<!fir.array<?xf32>>> cleanup {
        %1 = fir.box_addr %0 : !fir.box<!fir.heap<!fir.array<?xf32>>> -> !fir.heap<!fir.array<?xf32>>
        %fir.freemem %1 : !fir.heap<!fir.array<?xf32>>
      }
    }
    ```
  }];

  let arguments = (ins AnyFortranEntity:$entity);
  let regions = (region  MaxSizedRegion<1>:$cleanup);

  let assemblyFormat = "$entity attr-dict `:` type($entity) custom<YieldOpCleanup>($cleanup)";
}

def hlfir_ElementalAddrOp : hlfir_Op<"elemental_addr", [Terminator, HasParent<"RegionAssignOp">,
    RecursiveMemoryEffects, RecursivelySpeculatable, hlfir_ElementalOpInterface,
    AttrSizedOperandSegments]> {
  let summary = "Yield the address of a vector subscripted variable inside an hlfir.region_assign";
  let description = [{
    Special terminator node for the left-hand side region of an hlfir.region_assign
    to a vector subscripted entity.

    It represents how the address of an element of such entity is computed given
    one based indices.

    It is very similar to hlfir.elemental, except that it does not produce an SSA
    value because there is no hlfir type to describe a vector subscripted entity
    (the codegen of such type would be problematic). Hence, it is tightly linked
    to an hlfir.region_assign by its terminator property.

    An optional cleanup region may be provided if any of the subscript expressions
    of the designator require a cleanup.
    This allows documenting cleanups that cannot be generated after the vector
    subscripted designator usage (that has not been materizaled yet). The cleanups
    will be evaluated after the assignment once the related
    hlfir.region_assign is lowered.

    Example: "X(VECTOR) = Y"

    ```
    hlfir.region_assign {
      hlfir.yield %y : !fir.ref<!fir.array<20xf32>>
    } to {
      hlfir.elemental_addr %vector_shape  : !fir.shape<1> {
        ^bb0(%i: index):
        %0 = hlfir.designate %vector (%i)  : (!fir.ref<!fir.array<20xi32>>, index) -> !fir.ref<i32>
        %1 = fir.load %0 : !fir.ref<i32>
        %x_element_addr = hlfir.designate %x (%1)  : (!fir.ref<!fir.array<100xf32>>, i32) -> !fir.ref<f32>
        hlfir.yield %x_element_addr : !fir.ref<f32>
      }
    }
    ```
  }];

  let arguments = (ins
    fir_ShapeType:$shape,
    Optional<AnyPolymorphicObject>:$mold,
    Variadic<AnyIntegerType>:$typeparams,
    OptionalAttr<UnitAttr>:$unordered
  );

  let regions = (region  SizedRegion<1>:$body,
                         MaxSizedRegion<1>:$cleanup);

  let builders = [
    OpBuilder<(ins "mlir::Value":$shape,
          CArg<"mlir::Value", "{}">:$mold,
      CArg<"mlir::ValueRange", "{}">:$typeparams,
      CArg<"bool", "false">:$isUnordered)>
  ];

  let assemblyFormat = [{
    $shape (`mold` $mold^)? (`typeparams` $typeparams^)?
    (`unordered` $unordered^)?
    attr-dict `:` type(operands) $body
    custom<YieldOpCleanup>($cleanup)}];

  let extraClassDeclaration = [{
    mlir::Region::BlockArgListType getIndices() {
      return getBody().getArguments();
    }

    /// Return the hlfir::YieldOp terminator of the operation
    /// body. It yields the variable element address.
    /// This should only be called once the ElementalAddrOp has been built.
    hlfir::YieldOp getYieldOp();

    /// ElementalOpInterface implementation.

    mlir::Region& getElementalRegion() { return getBody(); }
    mlir::Value getElementEntity();
    mlir::Region* getElementCleanup();

    /// Must this elemental be evaluated in order?
    bool isOrdered() { return !getUnordered(); }
  }];

  let hasVerifier = 1;
}

/// Define ODS constraints to verify that a region ends with a yield of a
/// certain type.
def YieldIntegerOrEmpty : CPred<"yieldsIntegerOrEmpty($_self)">;
def YieldIntegerRegion : RegionConstraint<
  And<[SizedRegion<1>.predicate, YieldIntegerOrEmpty]>,
  "single block region that yields an integer scalar value">;
def MaybeYieldIntegerRegion : RegionConstraint<
  And<[MaxSizedRegion<1>.predicate, YieldIntegerOrEmpty]>,
  "optional single block region that yields an integer scalar value">;

def hlfir_ForallOp : hlfir_Op<"forall", [hlfir_OrderedAssignmentTreeOpInterface]> {
  let summary = "represent a Fortran forall";
  let description = [{
    This operation allows representing Fortran forall. It computes
    a set of "index-name" values based on lower bound, upper bound,
    and step values whose evaluations are represented in their own
    regions.

    Operations nested in its body region are evaluated in order.
    As opposed to a regular loop, each nested operation is
    fully evaluated for all the values in the "active set of
    index-name" before the next nested operation. In practice, the
    nested operation evaluation may be fused if it is proven that
    they do not have data dependency.

    The "index-name" value is represented as the argument of the
    body region.

    The lower, upper, and step region (if provided), must be terminated
    by hlfir.yield that yields scalar integers.

    The body region must only contain other OrderedAssignmentTreeOpInterface
    operations (like hlfir.region_assign, or other hlfir.forall).

    A Fortran forall with several indices is represented as a nest
    of hlfir.forall.

    All the regions contained in the hlfir.forall must only contain
    code that is pure from a Fortran point of view, except for the
    assignment effect of the hlfir.region_assign.
    This matches Fortran constraint C1037, but requires the outer
    controls to be evaluated outside of the hlfir.forall (these
    controls may have side effects as per Fortran 2018 10.1.4 section).

    Example: FORALL(I=1:10) X(I) = FOO(I)
    ```
      hlfir.forall lb {
        hlfir.yield %c1 : index
      } ub {
        hlfir.yield %c10 : index
      } (%i : index) {
        hlfir.region_assign {
          %res = fir.call @foo(%i) : (index) -> f32
          hlfir.yield %res : f32
        } to {
          %xi = hlfir.designate %x(%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
          hlfir.yield %xi : !fir.ref<f32>
        }
      }
    ```

  }];

  let regions = (region  YieldIntegerRegion:$lb_region,
                         YieldIntegerRegion:$ub_region,
                         MaybeYieldIntegerRegion:$step_region,
                         SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    mlir::Value getForallIndexValue() {
      return getBody().getArguments()[0];
    }
    void getLeafRegions(llvm::SmallVectorImpl<mlir::Region*>& regions) {
      regions.push_back(&getLbRegion());
      regions.push_back(&getUbRegion());
      if (!getStepRegion().empty())
        regions.push_back(&getStepRegion());
    }
    mlir::Region* getSubTreeRegion() { return &getBody(); }
  }];

  let assemblyFormat = [{
    attr-dict `lb` $lb_region
    `ub` $ub_region
    (`step` $step_region^)?
    custom<ForallOpBody>($body)
  }];
}

/// Shared definition for hlfir.forall_mask and hlfir.where
/// that have the same structure and assembly format, but not the same
/// constraints.
class hlfir_AssignmentMaskOp<string mnemonic> : hlfir_Op<mnemonic,
    [hlfir_OrderedAssignmentTreeOpInterface]> {
  let regions = (region  SizedRegion<1>:$mask_region,
                         SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    void getLeafRegions(llvm::SmallVectorImpl<mlir::Region*>& regions) {
      regions.push_back(&getMaskRegion());
    }
    mlir::Region* getSubTreeRegion() { return &getBody(); }
  }];

  let assemblyFormat = [{
    $mask_region
    attr-dict `do`
    custom<AssignmentMaskOpBody>($body)
  }];
}

def hlfir_ForallMaskOp : hlfir_AssignmentMaskOp<"forall_mask"> {
  let summary = "Represent a Fortran forall mask";
  let description = [{
    Fortran Forall can have a scalar mask expression that depends on the
    Forall index-name value.
    hlfir.forall_mask allows representing this mask. The expression
    evaluation is held in the mask region that must yield an i1 scalar
    value.
    An hlfir.forall_mask must be directly nested in the body region of
    an hlfir.forall. It is a separate operation so that it can use the
    index SSA value defined by the hlfir.forall body region.

    Example: "FORALL(I=1:10, SOME_CONDITION(I)) X(I) = FOO(I)"
    ```
    hlfir.forall lb {
      hlfir.yield %c1 : index
    } ub {
      hlfir.yield %c10 : index
    } (%i : index) {
      hlfir.forall_mask {
        %mask = fir.call @some_condition(%i) : (index) -> i1
        hlfir.yield %mask : i1
      } do {
        hlfir.region_assign {
          %res = fir.call @foo(%i) : (index) -> f32
          hlfir.yield %res : f32
        } to {
          %xi = hlfir.designate %x(%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
          hlfir.yield %xi : !fir.ref<f32>
        }
      }
    }
    ```
  }];
  let hasVerifier = 1;
}

def hlfir_WhereOp : hlfir_AssignmentMaskOp<"where"> {
  let summary = "Represent a Fortran where construct or statement";
  let description = [{
    Represent Fortran "where" construct or statement. The mask
    expression evaluation is held in the mask region that must yield
    logical array that has the same shape as all the nested
    hlfir.region_assign left-hand sides, and all the nested hlfir.where
    or hlfir.elsewhere masks.

    The values of the where and elsewhere masks form a control mask that
    controls all the nested hlfir.region_assign: only the array element for
    which the related control mask value is true are assigned. Any right-hand
    side elemental expression is only evaluated for elements where the control
    mask is true. See Fortran standard 2018 section 10.2.3 for more detailed
    about the control mask semantic.

    An hlfir.where must not contain any hlfir.forall but it may be contained
    in such operation. This matches Fortran rules.
  }];
  let hasVerifier = 1;
}

def hlfir_ElseWhereOp : hlfir_Op<"elsewhere", [Terminator,
    ParentOneOf<["WhereOp", "ElseWhereOp"]>, hlfir_OrderedAssignmentTreeOpInterface]> {
  let summary = "Represent a Fortran elsewhere statement";

  let description = [{
    Represent Fortran "elsewhere" construct or statement.

    It has an optional mask region to hold the evaluation of Fortran
    optional elsewhere mask expressions. If this region is provided,
    it must satisfy the same constraints as hlfir.where mask region.

    An hlfir.elsewhere must be the last operation of an hlfir.where or,
    hlfir.elsewhere body, which is enforced by its terminator property.

    Like in Fortran, an hlfir.elsewhere negate the current control mask,
    and if provided, adds the mask the resulting control mask (with a logical
    AND).
  }];

  let regions = (region  MaxSizedRegion<1>:$mask_region,
                         SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    void getLeafRegions(llvm::SmallVectorImpl<mlir::Region*>& regions) {
      if (!getMaskRegion().empty())
        regions.push_back(&getMaskRegion());
    }
    mlir::Region* getSubTreeRegion() { return &getBody(); }
  }];

  let assemblyFormat = [{
    (`mask` $mask_region^)?
    attr-dict `do`
    custom<AssignmentMaskOpBody>($body)
  }];
  let hasVerifier = 1;
}

def hlfir_ForallIndexOp : hlfir_Op<"forall_index", [fir_FortranVariableOpInterface,
    hlfir_OrderedAssignmentTreeOpInterface, Pure]> {
  let summary = "represent a Fortran forall index declaration";
  let description = [{
    This operation allows placing an hlfir.forall index in memory with
    the related Fortran index-value name and type.

    So far, lowering needs to manipulate symbols as memory entities.
    This operation allows fulfilling this requirements without allowing
    bare alloca/declare/store inside the body of hlfir.forall, which would
    make their analysis more complex.

    Given Forall index-value cannot be modified it also allows defining
    a canonicalization of all its loads into a fir.convert of the
    hlfir.forall index, which helps simplifying the data dependency analysis
    of hlfir.forall.
  }];

  let arguments = (ins AnyIntegerType:$index,
                       Builtin_StringAttr:$name);

  let results = (outs AnyFortranVariable);

  let assemblyFormat = [{
    $name $index attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    /// Implement FortranVariableInterface interface.
    std::optional<fir::FortranVariableFlagsEnum> getFortranAttrs() const {
      return std::nullopt;
    }
    mlir::Value getShape() const {return mlir::Value{};}
    mlir::OperandRange getExplicitTypeParams() const {
      // Return an empty range.
      return {(*this)->getOperands().begin(), (*this)->getOperands().begin()};
    }
    /// Implement OrderedAssignmentTreeOpInterface interface.
    void getLeafRegions(llvm::SmallVectorImpl<mlir::Region*>& regions) {}
    mlir::Region* getSubTreeRegion() { return nullptr; }
  }];

  let hasCanonicalizeMethod = 1;
}

def hlfir_CharExtremumOp : hlfir_Op<"char_extremum",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Find max/min from given character strings";
  let description = [{
    Find the lexicographical minimum or maximum of two or more character 
    strings of the same character kind and return the string with the lexicographical 
    minimum or maximum number of characters. Example:

    %0 = hlfir.char_extremum min, %arg0, %arg1 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,10>>
  }];

  let arguments = (ins hlfir_CharExtremumPredicateAttr:$predicate,
                  Variadic<AnyScalarCharacterEntity>:$strings
  );

  let results = (outs AnyScalarCharacterExpr);

  let assemblyFormat = [{
    $predicate `,` $strings attr-dict `:` functional-type(operands, results)
  }];

  let builders = [OpBuilder<(ins "hlfir::CharExtremumPredicate":$predicate, "mlir::ValueRange":$strings)>];

  let hasVerifier = 1;
}

#endif // FORTRAN_DIALECT_HLFIR_OPS
