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

def ACCEmitNYIFlang
    : Pass<"acc-emit-nyi-flang", "mlir::ModuleOp"> {
  let summary = "Emit not-yet-implemented messages for OpenACC";
  let description = [{
    Emits not-yet-implemented messages for OpenACC directives that are not yet
    supported through code generation.
  }];
}

def ACCInitializeFIRAnalyses
    : Pass<"acc-initialize-fir-analyses", "mlir::ModuleOp"> {
  let summary = "Initialize FIR analyses for OpenACC passes";
  let description = [{
    This pass initializes analyses that can be used by subsequent OpenACC passes
    in the pipeline. It creates and caches the OpenACCSupport analysis with a
    FIR-specific implementation that can handle FIR types and operations.
    It can also initialize FIR's AliasAnalysis for use in OpenACC passes.
    This pass needs to rerun if any analyses were invalidated by MLIR's framework.
  }];
  let options = [
    Option<"addFIRAliasAnalysis", "add-fir-alias-analysis", "bool",
           /*default=*/"true",
           "Initialize FIR AliasAnalysis for use by subsequent OpenACC passes.">
  ];
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

def ACCDeclareActionConversion
    : Pass<"acc-declare-action-conversion", "mlir::ModuleOp"> {
  let summary = "Convert acc declare action attributes to function calls";
  let description = [{
    Implements OpenACC specification (section 2.13.2 in OpenACC 3.4) for
    allocation/deallocation actions of allocatables and pointers in declare
    directives. Lowering generates recipe functions and marks the operations
    that allocate or deallocate the object; this pass finds those operations
    and inserts calls to the corresponding recipes.
  }];
  let dependentDialects = ["fir::FIROpsDialect"];
}

def ACCOptimizeFirstprivateMap
    : Pass<"acc-optimize-firstprivate-map", "mlir::func::FuncOp"> {
  let summary = "Optimize firstprivate mapping";
  let description = [{
    This pass optimizes acc firstprivate mapping operations by hoisting
    loads from the mapped variable to before the compute region. This enables
    pass-by-value instead of using global memory mapping through the
    runtime.
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect", "fir::FIROpsDialect"];
}

def ACCDevicePtrToCUFKernel
    : Pass<"acc-device-ptr-to-cuf-kernel", "mlir::ModuleOp"> {
  let summary = "Pass device addresses to CUDA Fortran kernels launched inside "
                "OpenACC data regions";
  let description = [{
    A CUDA Fortran kernel launched inside an OpenACC data region must receive
    the device address of any host variable that OpenACC has made present, not
    the host address. Otherwise the kernel dereferences a host pointer, which is
    only valid on shared-memory/unified-addressing systems and is illegal on a
    device with a separate address space.

    For each cuf.kernel_launch whose reference arguments resolve to a variable
    mapped by an enclosing acc.data region, this pass wraps the launch in an
    acc.host_data construct with acc.use_device operands for the mapped
    variables, and rebuilds the launch argument addressing on top of the
    use_device result so the kernel receives the device pointer.

    Both directly-addressed variables (e.g. static arrays, whose data address is
    the mapped varPtr) and descriptor-based variables (e.g. allocatables and
    pointers, whose data address is box_addr(load(<descriptor>))) are handled.

    This must run before the pass that lowers cuf.kernel_launch to
    gpu.launch_func.
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect", "fir::FIROpsDialect",
                           "cuf::CUFDialect"];
}

#endif // FORTRAN_OPTIMIZER_OPENACC_PASSES
