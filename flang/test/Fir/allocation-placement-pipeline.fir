// Test that the default optimizer pipeline uses the unified
// allocation-placement pass in place of the legacy stack-arrays /
// memory-allocation-opt passes, and that -enable-allocation-placement=false
// restores the legacy passes.

// RUN: tco %s --mlir-pass-statistics --mlir-pass-statistics-display=pipeline 2>&1 | FileCheck %s --check-prefix=NEW
// RUN: tco -enable-allocation-placement=false %s --mlir-pass-statistics --mlir-pass-statistics-display=pipeline 2>&1 | FileCheck %s --check-prefix=OLD

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
