//===-- CGOps.td - FIR operation definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the FIRCG dialect operations
///
//===----------------------------------------------------------------------===//

#ifndef FIR_DIALECT_CG_OPS
#define FIR_DIALECT_CG_OPS

include "mlir/IR/SymbolInterfaces.td"
include "flang/Optimizer/TypePredicates.td"

def fircg_Dialect : Dialect {
  let name = "fircg";
  let cppNamespace = "::fir::cg";
}

// Base class for FIR CG operations.
// All operations automatically get a prefix of "fircg.".
class fircg_Op<string mnemonic, list<OpTrait> traits>
  : Op<fircg_Dialect, mnemonic, traits>;

// Extended embox operation.
def fircg_XEmboxOp : fircg_Op<"ext_embox", [AttrSizedOperandSegments]> {
  let summary = "for internal conversion only";

  let description = [{
    Prior to lowering to LLVM IR dialect, a non-scalar non-trivial embox op will
    be converted to an extended embox. This op will have the following sets of
    arguments.

       - memref: The memory reference being emboxed.
       - shape: A vector that is the runtime shape of the underlying array.
       - shift: A vector that is the runtime origin of the first element.
         The default is a vector of the value 1.
       - slice: A vector of triples that describe an array slice.
       - subcomponent: A vector of indices for subobject slicing.
       - LEN type parameters: A vector of runtime LEN type parameters that
         describe an correspond to the elemental derived type.

    The memref and shape arguments are mandatory. The rest are optional.
  }];

  let arguments = (ins
    AnyReferenceLike:$memref,
    Variadic<AnyIntegerType>:$shape,
    Variadic<AnyIntegerType>:$shift,
    Variadic<AnyIntegerType>:$slice,
    Variadic<AnyCoordinateType>:$subcomponent,
    Variadic<AnyIntegerType>:$lenParams
  );
  let results = (outs fir_BoxType);

  let assemblyFormat = [{
    $memref (`(`$shape^`)`)? (`origin` $shift^)? (`[`$slice^`]`)? (`path` $subcomponent^)? (`typeparams` $lenParams^)? attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    unsigned getRank() { return shape().size(); }
    unsigned lenParamOffset() {
      return 1 + shape().size() + shift().size() + slice().size() +
        subcomponent().size();
    }
  }];
}

// Extended array coordinate operation.
def fircg_XArrayCoorOp : fircg_Op<"ext_array_coor", [AttrSizedOperandSegments]> {
  let summary = "for internal conversion only";

  let description = [{
    Prior to lowering to LLVM IR dialect, a non-scalar non-trivial embox op will
    be converted to an extended embox. This op will have the following sets of
    arguments.

       - memref: The memory reference of the array's data.
       - shape: A vector that is the runtime shape of the underlying array.
       - shift: A vector that is the runtime origin of the first element.
         The default is a vector of the value 1.
       - slice: A vector of triples that describe an array slice.
       - subcomponent: A vector of indices that describe subobject slicing.
       - indices: A vector of runtime values that describe the coordinate of
         the element of the array to be computed.
       - LEN type parameters: A vector of runtime LEN type parameters that
         describe an correspond to the elemental derived type.

    The memref, shape, and indices arguments are mandatory. The rest are
    optional.
  }];

  let arguments = (ins
    AnyReferenceLike:$memref,
    Variadic<AnyIntegerType>:$shape,
    Variadic<AnyIntegerType>:$shift,
    Variadic<AnyIntegerType>:$slice,
    Variadic<AnyCoordinateType>:$subcomponent,
    Variadic<AnyCoordinateType>:$indices,
    Variadic<AnyIntegerType>:$lenParams
  );
  let results = (outs fir_ReferenceType);

  let assemblyFormat = [{
    $memref (`(`$shape^`)`)? (`origin` $shift^)? (`[`$slice^`]`)? (`path` $subcomponent^)? `<`$indices`>` (`typeparams` $lenParams^)? attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    unsigned getRank() { return shape().size(); }
  }];
}

#endif
