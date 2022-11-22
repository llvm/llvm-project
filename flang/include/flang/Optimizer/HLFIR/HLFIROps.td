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

  let results = (outs AnyFortranVariableLike, AnyRefOrBoxLike);

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

#endif // FORTRAN_DIALECT_HLFIR_OPS
