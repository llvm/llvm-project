// Test that the default optimizer pipeline uses the unified
// allocation-placement pass in place of the legacy stack-arrays /
// memory-allocation-opt passes.

// RUN: tco %s --mlir-pass-statistics --mlir-pass-statistics-display=pipeline 2>&1 | FileCheck %s

// REQUIRES: asserts

// CHECK-NOT: MemoryAllocationOpt
// CHECK-NOT: StackArrays
// CHECK: AllocationPlacement

func.func @_QPfoo() {
  return
}
