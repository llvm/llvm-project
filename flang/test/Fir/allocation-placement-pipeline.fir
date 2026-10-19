// Test that the experimental -enable-allocation-placement flag replaces the
// legacy stack-arrays / memory-allocation-opt passes with the unified
// allocation-placement pass in the default optimizer pipeline. By default the
// legacy passes are still used.

// RUN: tco -enable-allocation-placement %s --mlir-pass-statistics --mlir-pass-statistics-display=pipeline 2>&1 | FileCheck %s --check-prefix=NEW
// RUN: tco %s --mlir-pass-statistics --mlir-pass-statistics-display=pipeline 2>&1 | FileCheck %s --check-prefix=OLD

// REQUIRES: asserts

// NEW-NOT: MemoryAllocationOpt
// NEW: AllocationPlacement
// NEW-NOT: MemoryAllocationOpt

// OLD-NOT: AllocationPlacement
// OLD: MemoryAllocationOpt
// OLD-NOT: AllocationPlacement

func.func @_QPfoo() {
  return
}
