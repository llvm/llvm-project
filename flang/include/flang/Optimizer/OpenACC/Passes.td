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

#endif // FORTRAN_OPTIMIZER_OPENACC_PASSES
