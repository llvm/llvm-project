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

#ifndef FORTRAN_DIALECT_FIRCG_OPS
#define FORTRAN_DIALECT_FIRCG_OPS

include "mlir/IR/SymbolInterfaces.td"
include "flang/Optimizer/Dialect/FIRTypes.td"

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
       - substring: A substring operator (offset, length) for CHARACTER.
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
    Variadic<AnyIntegerType>:$substr,
    Variadic<AnyIntegerType>:$lenParams
  );
  let results = (outs fir_BoxType);

  let assemblyFormat = [{
    $memref (`(`$shape^`)`)? (`origin` $shift^)? (`[`$slice^`]`)?
      (`path` $subcomponent^)? (`substr` $substr^)? (`typeparams` $lenParams^)?
      attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    // The rank of the entity being emboxed
    unsigned getRank() { return shape().size(); }

    // The rank of the result. A slice op can reduce the rank.
    unsigned getOutRank();

    // The shape operands are mandatory and always start at 1.
    unsigned shapeOffset() { return 1; }
    unsigned shiftOffset() { return shapeOffset() + shape().size(); }
    unsigned sliceOffset() { return shiftOffset() + shift().size(); }
    unsigned subcomponentOffset() { return sliceOffset() + slice().size(); }
    unsigned substrOffset() {
      return subcomponentOffset() + subcomponent().size();
    }
    unsigned lenParamOffset() { return substrOffset() + substr().size(); }
  }];
}

// Extended rebox operation.
def fircg_XReboxOp : fircg_Op<"ext_rebox", [AttrSizedOperandSegments]> {
  let summary = "for internal conversion only";

  let description = [{
    Prior to lowering to LLVM IR dialect, a non-scalar non-trivial rebox op will
    be converted to an extended rebox. This op will have the following sets of
    arguments.

       - box: The box being reboxed.
       - shape: A vector that is the new runtime shape for the array
       - shift: A vector that is the new runtime origin of the first element.
         The default is a vector of the value 1.
       - slice: A vector of triples that describe an array slice.
       - subcomponent: A vector of indices for subobject slicing.
       - substring: A substring operator (offset, length) for CHARACTER.

    The box argument is mandatory, the other arguments are optional.
    There must not both be a shape and slice/subcomponent arguments
  }];

  let arguments = (ins
    fir_BoxType:$box,
    Variadic<AnyIntegerType>:$shape,
    Variadic<AnyIntegerType>:$shift,
    Variadic<AnyIntegerType>:$slice,
    Variadic<AnyCoordinateType>:$subcomponent,
    Variadic<AnyIntegerType>:$substr
  );
  let results = (outs fir_BoxType);

  let assemblyFormat = [{
    $box (`(`$shape^`)`)? (`origin` $shift^)? (`[`$slice^`]`)?
      (`path` $subcomponent^)? (`substr` $substr^)? attr-dict `:`
      functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    // The rank of the entity being reboxed
    unsigned getRank();
    // The rank of the result box
    unsigned getOutRank();

    unsigned shapeOffset() { return 1; }
    unsigned shiftOffset() { return shapeOffset() + shape().size(); }
    unsigned sliceOffset() { return shiftOffset() + shift().size(); }
    unsigned subcomponentOffset() { return sliceOffset() + slice().size(); }
    unsigned substrOffset() {
      return subcomponentOffset() + subcomponent().size();
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

       - memref: The memory reference of the array's data. It can be a fir.box if
         the underlying data is not contiguous.
       - shape: A vector that is the runtime shape of the underlying array.
       - shift: A vector that is the runtime origin of the first element.
         The default is a vector of the value 1.
       - slice: A vector of triples that describe an array slice.
       - subcomponent: A vector of indices that describe subobject slicing.
       - indices: A vector of runtime values that describe the coordinate of
         the element of the array to be computed.
       - LEN type parameters: A vector of runtime LEN type parameters that
         describe an correspond to the elemental derived type.

    The memref and indices arguments are mandatory.
    The shape argument is mandatory if the memref is not a box, and should be
    omitted otherwise. The rest of the arguments are optional.
  }];

  let arguments = (ins
    AnyRefOrBox:$memref,
    Variadic<AnyIntegerType>:$shape,
    Variadic<AnyIntegerType>:$shift,
    Variadic<AnyIntegerType>:$slice,
    Variadic<AnyCoordinateType>:$subcomponent,
    Variadic<AnyCoordinateType>:$indices,
    Variadic<AnyIntegerType>:$lenParams
  );
  let results = (outs fir_ReferenceType);

  let assemblyFormat = [{
    $memref (`(`$shape^`)`)? (`origin` $shift^)? (`[`$slice^`]`)?
      (`path` $subcomponent^)? `<`$indices`>` (`typeparams` $lenParams^)?
      attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    unsigned getRank();

    // Shape is optional, but if it exists, it will be at offset 1.
    unsigned shapeOffset() { return 1; }
    unsigned shiftOffset() { return shapeOffset() + shape().size(); }
    unsigned sliceOffset() { return shiftOffset() + shift().size(); }
    unsigned subcomponentOffset() { return sliceOffset() + slice().size(); }
    unsigned indicesOffset() {
      return subcomponentOffset() + subcomponent().size();
    }
    unsigned lenParamsOffset() { return indicesOffset() + indices().size(); }
  }];
}

#endif
