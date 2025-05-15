//===-- PRIFOps.td - PRIF operation definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the PRIF dialect operations
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_DIALECT_PRIF_PRIF_OPS
#define FORTRAN_DIALECT_PRIF_PRIF_OPS

include "flang/Optimizer/Dialect/PRIF/PRIFDialect.td"
include "flang/Optimizer/Dialect/FIRTypes.td"
include "flang/Optimizer/Dialect/FIRAttr.td"
include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/BuiltinAttributes.td"

class prif_Op<string mnemonic, list<Trait> traits>
    : Op<PRIFDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Initialization and Finalization
//===----------------------------------------------------------------------===//

def prif_InitOp : prif_Op<"init", []> {
  let summary = "Initialize the parallel environment";
  let description = [{This procedure will initialize the parallel environment}];

  let arguments = (ins I32:$stat);
}

def prif_StopOp : prif_Op<"stop", [AttrSizedOperandSegments]> {
  let summary = "Procedure that synchronizes all execution images, clean up the"
                "parallel runtime environment, and terminates the program.";
  let description = [{
    This procedure synchronizes all executing images, cleans up the parallel 
    runtime environment, and terminates the program. Calls to this procedure do 
    not return. This procedure supports both normal termination at the end of a
    program, as well as any STOP statements from the user source code.
  }];

  let arguments = (ins AnyReferenceLike:$quiet,
                   Optional<AnyReferenceLike>:$stop_code_int,
                   Optional<AnyRefOrBoxType>:$stop_code_char);

  let hasVerifier = 1;
}

def prif_ErrorStopOp : prif_Op<"error_stop", [AttrSizedOperandSegments]> {
  let summary = "This procedure terminates all executing images." 
                "Calls to this procedure do not return";
  let description = [{
    This procedure terminates all executing images and calls to this procedure 
    do not return.
  }];

  let arguments = (ins AnyReferenceLike:$quiet,
                   Optional<AnyReferenceLike>:$stop_code_int,
                   Optional<AnyRefOrBoxType>:$stop_code_char);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Image Queries
//===----------------------------------------------------------------------===//

def prif_NumImagesOp : prif_Op<"num_images", [AttrSizedOperandSegments]> {
  let summary = "Query the number of images in the specified or current team";

  let arguments = (ins Optional<AnyRefOrBoxType>:$team_number, 
                       Optional<AnyReferenceLike>:$team);
  let results = (outs I32);
  
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins CArg<"mlir::Value", "{}">:$team_number,
                   CArg<"mlir::Value", "{}">:$team),
    [{ return build($_builder, $_state, team_number, team); }]>
  ];
  
  let hasVerifier = 1;
}

def prif_ThisImageOp : prif_Op<"this_image", []> {
  let summary = "Determine the image index of the current image";

  let arguments = (ins Optional<AnyRefOrBoxType>:$team);
  let results = (outs I32);
}

//===----------------------------------------------------------------------===//
// Coarray Queries
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Allocation and Deallocation
//===----------------------------------------------------------------------===//

def prif_AllocateCoarrayOp : prif_Op<"allocate_coarray",
    [MemoryEffects<[MemAlloc<DefaultResource>]>]> {
  let summary = "Perform the allocation of a coarray and provide a "
                "corresponding coarray descriptor";

  let description = [{
    This procedure allocates a coarray and provides a handle referencing a
    corresponding coarray descriptor. This call is collective over the
    current team.
  }];

  let arguments = (ins Arg<AnyRefOrBoxType, "", [MemRead]>:$lcobounds,
                       Arg<AnyRefOrBoxType, "", [MemRead]>:$ucobound,
                       Arg<AnyReferenceLike, "", [MemRead]>:$size_in_bytes,
                       Arg<FunctionType, "", [MemRead]>:$final_func,
                       Arg<AnyRefOrBoxType, "", [MemWrite]>:$coarray_handle,
                       Arg<AnyRefOrBoxType, "", [MemWrite]>:$allocated_memory,
                       Arg<Optional<AnyRefOrBoxType>, "", [MemWrite]>:$errmsg);

  let results = (outs I32:$stat);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def prif_SyncAllOp : prif_Op<"sync_all", [AttrSizedOperandSegments]> {
  let summary =
      "Performs a collective synchronization of all images in the current team";

  let arguments = (ins Optional<AnyRefOrBoxType>:$errmsg);
  
  let results = (outs I32:$stat);
  
  let hasVerifier = 1;
}

#endif // FORTRAN_DIALECT_PRIF_PRIF_OPS
