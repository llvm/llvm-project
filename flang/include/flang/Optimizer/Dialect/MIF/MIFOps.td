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

class region_Op<string mnemonic, list<Trait> traits = []>
    : mif_Op<mnemonic, !listconcat(traits, [RecursivelySpeculatable,
                                            RecursiveMemoryEffects])> {}

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

def mif_SyncTeamOp : mif_Op<"sync_team", [AttrSizedOperandSegments]> {
  let summary = "Performs a synchronization of the team, identified by `team`";

  let arguments = (ins AnyRefOrBoxType:$team, Optional<AnyReferenceLike>:$stat,
      Optional<AnyRefOrBoxType>:$errmsg);
  let assemblyFormat = [{
    $team (`stat` $stat^ )?
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

//===----------------------------------------------------------------------===//
// Teams
//===----------------------------------------------------------------------===//

def mif_FormTeamOp : mif_Op<"form_team", [AttrSizedOperandSegments]> {
  let summary =
      "Create a set of sibling teams whose parent team is the current team.";
  let description = [{
    Create a new team for each unique `team_number` value specified.
    Each executing image will belong to the team whose `team_number` is equal 
    to the value of team-number on that image, and `team_var` becomes defined
    with a value that identifies that team.

    If `new_index` is specified, the image index of the executing image will take
    this index in its new team. Otherwise, the new image index is processor 
    dependent.

    Arguments: 
      - `team_number`: Shall be a positive integer.
      - `team_var` : Shall be a variable of type TEAM_TYPE from the intrinsic
        module ISO_FORTRAN_ENV.
      - `new_index`(optional): Shall be an integer that correspond to the index that
        the calling image will have in the new team.
  }];

  let arguments = (ins AnyIntegerType:$team_number,
      Arg<fir_BoxType, "", [MemWrite]>:$team_var,
      Optional<AnyIntegerType>:$new_index,
      Arg<Optional<AnyReferenceLike>, "", [MemWrite]>:$stat,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);

  let assemblyFormat = [{
    `team_number` $team_number `team_var` $team_var
    (`new_index` $new_index^ )?
    (`stat` $stat^ )?
    (`errmsg` $errmsg^ )?
    attr-dict `:` functional-type(operands, results)
  }];
}

def mif_EndTeamOp : mif_Op<"end_team", [AttrSizedOperandSegments, Terminator,
                                        ParentOneOf<["ChangeTeamOp"]>]> {
  let summary = "Changes the current team to the parent team.";
  let description = [{
    The END TEAM operation completes the CHANGE TEAM construct and 
    restores the current team to the team that was current before 
    the CHANGE TEAM construct. 
  }];

  let arguments = (ins Arg<Optional<AnyReferenceLike>, "", [MemWrite]>:$stat,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let builders = [OpBuilder<(ins), [{ /* do nothing */ }]>];

  let assemblyFormat = [{
    (`stat` $stat^ )? (`errmsg` $errmsg^ )?
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// NOTE: The CHANGE TEAM region will take a coarray association list in
// argument. However, coarray management and coarray alias creation are not
// yet supported by the dialect. The argument is therefore not yet supported by
// this operation and will be added later.
//===----------------------------------------------------------------------===//
def mif_ChangeTeamOp
    : region_Op<"change_team", [AttrSizedOperandSegments,
                                SingleBlockImplicitTerminator<"EndTeamOp">]> {
  let summary = "Changes the current team.";
  let description = [{
    The CHANGE TEAM construct changes the current team to the specified new
    team, which must be a child team of the current team.  

    ```
      mif.change_team %team {
        %x = fir.convert %i : (index) -> i32
        ...
        mif.end_team
      }
  }];

  let arguments = (ins AnyRefOrBoxType:$team,
      Arg<Optional<AnyReferenceLike>, "", [MemWrite]>:$stat,
      Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders =
      [OpBuilder<(ins "mlir::Value":$team,
           CArg<"bool", "true">:$ensureTermination,
           CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes)>,
       OpBuilder<(ins "mlir::Value":$team, "mlir::Value":$stat,
           "mlir::Value":$errmsg, CArg<"bool", "true">:$ensureTermination,
           CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes)>];

  let extraClassDeclaration = [{
    /// Get the body of the CHANGE TEAM construct 
    mlir::Block *getBody() { return &getRegion().front(); }
  }];

  let assemblyFormat = [{
    $team (`stat` $stat^)?
    (`errmsg` $errmsg^)?
    attr-dict `:` `(` type(operands) `)`
    custom<ChangeTeamOpBody>($region)
  }];
}

def mif_GetTeamOp : mif_Op<"get_team", []> {
  let summary = "Get the team value for the current or ancestor team.";
  let description = [{
    This operation gets the team value for the current or an ancestor team.
    `level`(optional): If provided, must equal one of the following constants : 
    `INITIAL_TEAM`, `PARENT_TEAM` or `CURRENT_TEAM` from the module ISO_FORTRAN_ENV.
    If `level` isn't present or has the value `CURRENT_TEAM` the returned
    value is the current team.     
  }];

  let arguments = (ins Optional<AnyIntegerType>:$level);
  let results = (outs fir_BoxType:$team);

  let assemblyFormat = [{
    (`level` $level^ )?
    attr-dict `:` functional-type(operands, results)
  }];
}

def mif_TeamNumberOp : mif_Op<"team_number", []> {
  let summary = "Get the team number";
  let description = [{
    Argument: `team` is optional and shall be a scalar of type TEAM_TYPE from 
    module ISO_FORTRAN_ENV and the value identifies the current or an ancestor team. 
    If `team` is absent, the team specified is the current team.
  }];

  let arguments = (ins Optional<AnyRefOrBoxType>:$team);
  let results = (outs I64);

  let assemblyFormat = [{
    (`team` $team^ )?
    attr-dict `:` functional-type(operands, results)
  }];
}

#endif // FORTRAN_DIALECT_MIF_MIF_OPS
