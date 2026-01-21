//===-- Passes.td - flang OpenACC pass definitions -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_OPENACC_PASSES
#define FORTRAN_OPTIMIZER_OPENACC_PASSES

include "mlir/Pass/PassBase.td"

def ACCInitializeFIRAnalyses
    : Pass<"acc-initialize-fir-analyses", "mlir::ModuleOp"> {
  let summary = "Initialize FIR analyses for OpenACC passes";
  let description = [{
    This pass initializes analyses that can be used by subsequent OpenACC passes
    in the pipeline. It creates and caches the OpenACCSupport analysis with a
    FIR-specific implementation that can handle FIR types and operations.
    It also initializes FIR's AliasAnalysis for use in OpenACC passes.
    This pass needs to rerun if any analyses were invalidated by MLIR's framework.
  }];
  // In addition to pre-registering the needed analyses, this pass also
  // pre-registers the dialects that various OpenACC passes may generate.
  let dependentDialects = ["fir::FIROpsDialect", "hlfir::hlfirDialect",
      "mlir::acc::OpenACCDialect"];
}

def ACCRecipeBufferization
    : Pass<"fir-acc-recipe-bufferization", "mlir::ModuleOp"> {
  let summary = "Rewrite acc.*.recipe box values to ref<box> and update uses";
  let description = [{
    Bufferizes OpenACC recipes that operate on fir.box<T> so their type and
    region block arguments become fir.ref<fir.box<T>> instead. This applies to
    acc.private.recipe, acc.firstprivate.recipe (including copy region), and
    acc.reduction.recipe (including combiner region).

    For affected regions, the pass inserts required loads at the beginning of
    the region to preserve original uses after argument type changes. For yields
    of box values, the pass allocates a local fir.ref<fir.box<T>> and stores the
    yielded fir.box<T> into it so the region yields a reference to a box.

    For acc.private, acc.firstprivate, and acc.reduction operations that use a
    bufferized recipe, the pass allocates a host-side fir.ref<fir.box<T>> before
    the data op and rewires the data op to use the new memory. Other users of
    the original data operation result (outside the paired compute op) are
    updated to load through the reference.
  }];
}

def ACCUseDeviceCanonicalizer
    : Pass<"acc-use-device-canonicalizer", "mlir::func::FuncOp"> {
  let summary = "Canonicalize acc.use_device operations for FIR box types";
  let description = [{
    This pass canonicalizes the use_device clause on a host_data construct such
    that use_device(x) can be lowered to a simple runtime call that takes the
    actual host pointer as argument.

    For a use_device operand that is a box type or a reference to a box, the
    pass:
      1. Extracts the host base address for mapping to a device address using
         acc.use_device.
      2. Creates a new boxed descriptor with the device address as the base
         address for use inside the host_data region.

    The pass also removes unused use_device clauses, reducing the number of
    runtime calls.
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect", "fir::FIROpsDialect"];
}

#endif // FORTRAN_OPTIMIZER_OPENACC_PASSES
