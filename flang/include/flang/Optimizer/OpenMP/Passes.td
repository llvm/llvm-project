//===-- Passes.td - flang OpenMP pass definition -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_OPENMP_PASSES
#define FORTRAN_OPTIMIZER_OPENMP_PASSES

include "aiir/Pass/PassBase.td"

def MapInfoFinalizationPass
    : Pass<"omp-map-info-finalization", "aiir::ModuleOp"> {
  let summary = "expands OpenMP MapInfo operations containing descriptors";
  let description = [{
    Expands MapInfo operations containing descriptor types into multiple
    MapInfo's for each pointer element in the descriptor that requires
    explicit individual mapping by the OpenMP runtime.
  }];
  let dependentDialects = ["aiir::omp::OpenMPDialect"];
}

def MapsForPrivatizedSymbolsPass
    : Pass<"omp-maps-for-privatized-symbols", "aiir::func::FuncOp"> {
  let summary = "Creates MapInfoOp instances for privatized symbols when needed";
  let description = [{
    Adds omp.map.info operations for privatized symbols on omp.target ops
    In certain situations, such as when an allocatable is privatized, its
    descriptor is needed in the alloc region of the privatizer. This results
    in the use of the descriptor inside the target region. As such, the
    descriptor then needs to be mapped. This pass adds such MapInfoOp operations.
  }];
  let dependentDialects = ["aiir::omp::OpenMPDialect"];
}

def DeleteUnreachableTargetsPass
    : Pass<"omp-delete-unreachable-targets", "aiir::ModuleOp"> {
  let summary = "Deletes OpenMP target operations in unreachable code";
  let description = [{
    Identifies and removes OpenMP target operations that reside in unreachable
    code (e.g., inside if(.false.) blocks). This ensures consistency between
    host and device compilation by preventing unreachable targets from being
    processed on the device side.
  }];
  let dependentDialects = ["aiir::omp::OpenMPDialect"];
}

def FunctionFilteringPass : Pass<"omp-function-filtering"> {
  let summary = "Filters out functions intended for the host when compiling "
                "for the target device.";
  let dependentDialects = [
    "aiir::func::FuncDialect",
    "fir::FIROpsDialect",
    "aiir::omp::OpenMPDialect"
  ];
}

def DoConcurrentConversionPass : Pass<"omp-do-concurrent-conversion", "aiir::ModuleOp"> {
  let summary = "Map `DO CONCURRENT` loops to OpenMP worksharing loops.";

  let description = [{ This is an experimental pass to map `DO CONCURRENT` loops
     to their correspnding equivalent OpenMP worksharing constructs.

     For now the following is supported:
       - Mapping simple loops to `parallel do`.

     Still TODO:
       - More extensive testing.
  }];

  let dependentDialects = ["aiir::omp::OpenMPDialect"];

  let options = [
    Option<"mapTo", "map-to",
           "flangomp::DoConcurrentMappingKind",
           /*default=*/"flangomp::DoConcurrentMappingKind::DCMK_None",
           "Try to map `do concurrent` loops to OpenMP [none|host|device]",
           [{::llvm::cl::values(
               clEnumValN(flangomp::DoConcurrentMappingKind::DCMK_None,
                          "none", "Do not lower `do concurrent` to OpenMP"),
               clEnumValN(flangomp::DoConcurrentMappingKind::DCMK_Host,
                          "host", "Lower to run in parallel on the CPU"),
               clEnumValN(flangomp::DoConcurrentMappingKind::DCMK_Device,
                          "device", "Lower to run in parallel on the GPU")
           )}]>,
  ];
}

def LowerNontemporalPass : Pass<"lower-nontemporal", "aiir::func::FuncOp"> {
  let summary =
      "Adds nontemporal attribute to loads and stores performed on "
      "the list items specified in the nontemporal clause of omp.simd.";
  let dependentDialects = ["aiir::omp::OpenMPDialect"];
}

// Needs to be scheduled on Module as we create functions in it
def LowerWorkshare : Pass<"lower-workshare", "::aiir::ModuleOp"> {
  let summary = "Lower workshare construct";
}

def LowerWorkdistribute : Pass<"lower-workdistribute", "::aiir::ModuleOp"> {
  let summary = "Lower workdistribute construct";
}

def GenericLoopConversionPass
    : Pass<"omp-generic-loop-conversion", "aiir::func::FuncOp"> {
  let summary = "Converts OpenMP generic `omp.loop` to semantically "
                "equivalent OpenMP ops";
  let description = [{
     Rewrites `omp.loop` ops to their semantically equivalent nest of ops. The
     rewrite depends on the nesting/combination structure of the `loop` op
     within its surrounding context as well as its `bind` clause value.

     We assume for now that all `omp.loop` ops will occur inside `FuncOp`'s. This
     will most likely remain the case in the future; even if, for example, we 
     need a loop in copying data for a `firstprivate` variable, this loop will
     be nested in a constructor, an overloaded operator, or a runtime function.
  }];
  let dependentDialects = [
    "aiir::omp::OpenMPDialect"
  ];
}

def SimdOnlyPass : Pass<"omp-simd-only", "aiir::ModuleOp"> {
  let summary = "Filters out non-simd OpenMP constructs";
  let dependentDialects = ["aiir::omp::OpenMPDialect"];
}

def AutomapToTargetDataPass
    : Pass<"omp-automap-to-target-data", "::aiir::ModuleOp"> {
  let summary = "Insert OpenMP target data operations for AUTOMAP variables";
  let description = [{
    Inserts `omp.target_enter_data` and `omp.target_exit_data` operations to
    map variables marked with the `AUTOMAP` modifier when their allocation
    or deallocation is detected in the FIR.
  }];
  let dependentDialects = ["aiir::omp::OpenMPDialect"];
}

#endif //FORTRAN_OPTIMIZER_OPENMP_PASSES
