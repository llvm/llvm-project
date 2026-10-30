// Test that a function opting out of -fstack-arrays in its own policy keeps its
// heap allocations, while a function without a policy of its own is unaffected.

// RUN: fir-opt --stack-arrays %s | FileCheck %s

// CHECK-LABEL: func.func @opted_out
// CHECK: fir.allocmem !fir.array<42xi32>
// CHECK-NOT: fir.alloca
func.func @opted_out() attributes {fir.allocation_policy =
                       #fir.allocation_policy<stack_arrays = false,
                           small_array_threshold = 1024,
                           total_stack_limit = 4194304>} {
  %0 = fir.allocmem !fir.array<42xi32>
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  %r = fir.convert %0 : (!fir.heap<!fir.array<42xi32>>) -> !fir.ref<!fir.array<42xi32>>
  %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
  fir.store %v to %e : !fir.ref<i32>
  fir.freemem %0 : !fir.heap<!fir.array<42xi32>>
  return
}

// CHECK-LABEL: func.func @no_policy
// CHECK: fir.alloca !fir.array<42xi32>
// CHECK-NOT: fir.allocmem
func.func @no_policy() {
  %0 = fir.allocmem !fir.array<42xi32>
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  %r = fir.convert %0 : (!fir.heap<!fir.array<42xi32>>) -> !fir.ref<!fir.array<42xi32>>
  %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
  fir.store %v to %e : !fir.ref<i32>
  fir.freemem %0 : !fir.heap<!fir.array<42xi32>>
  return
}
