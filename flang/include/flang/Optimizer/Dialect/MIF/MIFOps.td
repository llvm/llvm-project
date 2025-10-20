//===-- MIFOps.td - MIF operation definitions --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the MIF dialect operations
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_MIF_MIF_OPS
#define FORTRAN_DIALECT_MIF_MIF_OPS

include "flang/Optimizer/Dialect/MIF/MIFDialect.td"
include "flang/Optimizer/Dialect/FIRTypes.td"
include "flang/Optimizer/Dialect/FIRAttr.td"

class mif_Op<string mnemonic, list<Trait> traits>
    : Op<MIFDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Initialization and Finalization
//===----------------------------------------------------------------------===//

def mif_InitOp : mif_Op<"init", []> {
  let summary = "Initialize the parallel environment";
  let description = [{This operation will initialize the parallel environment}];

  let results = (outs I32:$stat);
  let assemblyFormat = "`->` type($stat) attr-dict";
}

//===----------------------------------------------------------------------===//
// Image Queries
//===----------------------------------------------------------------------===//

def mif_NumImagesOp : mif_Op<"num_images", [AttrSizedOperandSegments]> {
  let summary = "Query the number of images in the specified or current team";
  let description = [{
    This operation query the number of images in the specified or current
    team and can be called with 3 differents way :
    - `num_images()`
    - `num_images(team)`
    - `num_images(team_number)`

    Arguments:
    - `team` : Shall be a scalar of type `team_type` from the `ISO_FORTRAN_ENV`
            module with a value that identifies the current or ancestor team.
    - `team_number` :  Shall be an integer scalar. It shall identify the
            initial team or a sibling team of the current team.

    Result Value: The number of images in the specified team, or in the current
    team if no team is specified.
  }];

  let arguments = (ins Optional<AnyInteger>:$team_number,
                       Optional<AnyRefOrBoxType>:$team);
  let results = (outs I32:$res);

  let builders = [OpBuilder<(ins CArg<"mlir::Value", "{}">:$teamArg)>];

  let hasVerifier = 1;
  let assemblyFormat = [{
    ( `team_number` $team_number^ )? 
    ( `team` $team^ )? 
    attr-dict `:` functional-type(operands, results)
  }];
}

def mif_ThisImageOp : mif_Op<"this_image", [AttrSizedOperandSegments]> {
  let summary = "Determine the image index of the current image";
  let description = [{
    Arguments:
    - `coarray` :  Shall be a coarray of any type.
    - `dim` : Shall be an integer scalar. Its value shall be in the range of
          1 <= DIM <= N, where N is the corank of the coarray.
    - `team`(optional) : Shall be a scalar of type `team_type` from
          ISO_FORTRAN_ENV. If the `coarray` is present, it shall be
          established in that team.

    Results:
    - Case(1) : The result of `this_image([team])` is a scalar with a value
          equal to the index of the image in the current or specified team.
    - Case(2) : The result of `this_image(coarray [,team])` is the sequence of
          cosubscript values for `coarray`.
    - Case(3) : The result of `this_image(coarray, dim [,team])` is the value of
          cosubscript `dim` in the sequence of cosubscript values for `coarray`.

    Example:
    ```fortran
      REAL :: A[10, 0:9, 0:*]
    ```
    If we take a look on the example and we are on image 5, `this_image` has the
    value 5, `this_image(A)` has the value [5, 0, 0].
  }];

  let arguments = (ins Optional<fir_BoxType>:$coarray,
      Optional<AnyInteger>:$dim, Optional<AnyRefOrBoxType>:$team);
  let results = (outs I32:$res);

  let builders = [OpBuilder<(ins "mlir::Value":$coarray, "mlir::Value":$team)>,
                  OpBuilder<(ins "mlir::Value":$team)>];

  let hasVerifier = 1;
  let assemblyFormat = [{
    ( `coarray` $coarray^ )? 
    ( `team` $team^ )? 
    ( `dim` $dim^ )? 
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//
// NOTE: Every operation in this section corresponds to a Fortran 
//   "image control statement" (F23 11.7.1). This means they end a Fortran 
//   "segment" and start another (F23 11.7.2), which affects the memory 
//   consistency semantics of surrounding accesses in the multi-image execution. 
//   For now we've modeled this by *deliberately* omitting mlir::MemoryEffects
//   on these operations, to ensure analysis treats these ops conservatively.
//===----------------------------------------------------------------------===//

def mif_SyncAllOp : mif_Op<"sync_all", [AttrSizedOperandSegments]> {
  let summary =
      "Performs a collective synchronization of all images in the current team";

  let arguments = (ins Optional<AnyReferenceLike>:$stat,
                       Optional<AnyRefOrBoxType>:$errmsg);
  let assemblyFormat = [{
    (`stat` $stat^ )?
    (`errmsg` $errmsg^ )? 
    attr-dict `:` functional-type(operands, results)
  }];
}

def mif_SyncImagesOp : mif_Op<"sync_images", [AttrSizedOperandSegments]> {
  let summary = "Performs a synchronization of image with each of the other "
                "images in the `image_set`";
  let description = [{
    This operation can take an optional argument `⁣image_set`, wich must be an integer expression
    and must be scalar or rank one. If `image_set` is omitted from the call, this operation will 
    adopt the behavior of the Fortran statement `SYNC IMAGES(*)`.
  }];

  let arguments = (ins Optional<AnyRefOrBoxType>:$image_set,
                       Optional<AnyReferenceLike>:$stat,
                       Optional<AnyRefOrBoxType>:$errmsg);
  let hasVerifier = 1;
  let assemblyFormat = [{
    (`image_set` $image_set^ )? 
    (`stat` $stat^ )?
    (`errmsg` $errmsg^ )? 
    attr-dict `:` functional-type(operands, results)
  }];
}

def mif_SyncMemoryOp : mif_Op<"sync_memory", [AttrSizedOperandSegments]> {
  let summary = "Operation that ends one segment and begins another.";
  let description = [{
    Operation that ends one segment and begins another; Those two segments can 
    be ordered by user-defined way with respect to segments on other images.
  }];

  let arguments = (ins Optional<AnyReferenceLike>:$stat,
                       Optional<AnyRefOrBoxType>:$errmsg);
  let assemblyFormat = [{
    (`stat` $stat^ )?
    (`errmsg` $errmsg^ )? 
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Collective Operations
//===----------------------------------------------------------------------===//

def mif_CoBroadcastOp : mif_Op<"co_broadcast", [AttrSizedOperandSegments,
                                                MemoryEffects<[MemWrite]>]> {
  let summary = "Broadcast value to images.";
  let description = [{
    The co_broadcast operation broadcasts a value from one image to the other images.
  }];

  let arguments = (ins Arg<fir_BoxType, "", [MemRead, MemWrite]>:$a,
      AnyIntegerType:$source_image,
      Arg<Optional<AnyReferenceLike>, "", [MemWrite]>:$stat,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);

  let hasVerifier = 1;
  let assemblyFormat = [{
    $a `source` $source_image
    (`stat`  $stat^ )?
    (`errmsg` $errmsg^ )? 
    attr-dict `:` `(` type(operands) `)`
  }];
}

def mif_CoMaxOp
    : mif_Op<"co_max", [AttrSizedOperandSegments, MemoryEffects<[MemWrite]>]> {
  let summary = "Compute maximum value across images.";
  let description = [{
    The co_max operation performs the computation of the maximum 
    across images.
  }];

  let arguments = (ins Arg<fir_BoxType, "", [MemRead, MemWrite]>:$a,
      Optional<AnyIntegerType>:$result_image,
      Arg<Optional<AnyReferenceLike>, "", [MemWrite]>:$stat,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);

  let hasVerifier = 1;
  let assemblyFormat = [{
    $a (`result` $result_image^ )? 
    (`stat` $stat^ )?
    (`errmsg` $errmsg^ )? 
    attr-dict `:` `(` type(operands) `)`
  }];
}

def mif_CoMinOp
    : mif_Op<"co_min", [AttrSizedOperandSegments, MemoryEffects<[MemWrite]>]> {
  let summary = "Compute minimum value across images.";
  let description = [{
    The co_min operation performs the computation of the minimum
    across images.
  }];

  let arguments = (ins Arg<fir_BoxType, "", [MemRead, MemWrite]>:$a,
      Optional<AnyIntegerType>:$result_image,
      Arg<Optional<AnyReferenceLike>, "", [MemWrite]>:$stat,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);

  let hasVerifier = 1;
  let assemblyFormat = [{
    $a (`result` $result_image^ )? 
    (`stat` $stat^ )?
    (`errmsg` $errmsg^ )? 
    attr-dict `:` `(` type(operands) `)`
  }];
}

def mif_CoSumOp
    : mif_Op<"co_sum", [AttrSizedOperandSegments, MemoryEffects<[MemWrite]>]> {
  let summary = "Compute sum across images.";
  let description = [{
    The co_sum operation performs the computation of the sum
    across images.
  }];

  let arguments = (ins Arg<fir_BoxType, "", [MemRead, MemWrite]>:$a,
      Optional<AnyIntegerType>:$result_image,
      Arg<Optional<AnyReferenceLike>, "", [MemWrite]>:$stat,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);

  let hasVerifier = 1;
  let assemblyFormat = [{
    $a (`result` $result_image^ )? 
    (`stat` $stat^ )?
    (`errmsg` $errmsg^ )? 
    attr-dict `:` `(` type(operands) `)`
  }];
}

#endif // FORTRAN_DIALECT_MIF_MIF_OPS
