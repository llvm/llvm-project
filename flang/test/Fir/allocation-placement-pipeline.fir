// Test that the default optimizer pipeline uses the allocation-placement pass
// instead of the stack-arrays and memory-allocation-opt passes. With the
// experimental -enable-allocation-placement flag, it uses the size-based
// policy. Otherwise, it emulates the legacy passes.

// RUN: tco -enable-allocation-placement %s --mlir-pass-statistics --mlir-pass-statistics-display=pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=PASSES
// RUN: tco %s --mlir-pass-statistics --mlir-pass-statistics-display=pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=PASSES
// RUN: tco -enable-allocation-placement %s | FileCheck %s --check-prefix=NEW
// RUN: tco %s | FileCheck %s --check-prefix=OLD

// REQUIRES: asserts

// PASSES-NOT: MemoryAllocationOpt
// PASSES-NOT: StackArrays
// PASSES: AllocationPlacement
// PASSES-NOT: MemoryAllocationOpt
// PASSES-NOT: StackArrays

// NEW-LABEL: define void @small_temp
// NEW: alloca [10 x i32]
// NEW-NOT: @malloc

// OLD-LABEL: define void @small_temp
// OLD: call ptr @malloc
// OLD: call void @free

func.func private @use(!fir.ref<!fir.array<10xi32>>)

func.func @small_temp() {
  %0 = fir.allocmem !fir.array<10xi32>
  %1 = fir.convert %0 : (!fir.heap<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  fir.call @use(%1) : (!fir.ref<!fir.array<10xi32>>) -> ()
  fir.freemem %0 : !fir.heap<!fir.array<10xi32>>
  return
}
