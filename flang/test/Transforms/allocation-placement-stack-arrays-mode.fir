// Test the aggressive (-fstack-arrays) allocation-placement mode: all array
// temporaries are moved to the stack regardless of size, as long as the
// heap-to-stack conversion is provably safe.

// RUN: fir-opt --allocation-placement="stack-arrays=true" %s | FileCheck %s

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {

// Big temporary heap allocation -> stack under -fstack-arrays.
// CHECK-LABEL: func.func @big_temp_to_stack
// CHECK: fir.alloca !fir.array<100xi32>
// CHECK-NOT: fir.allocmem
func.func @big_temp_to_stack() {
  %0 = fir.allocmem !fir.array<100xi32>
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  %r = fir.convert %0 : (!fir.heap<!fir.array<100xi32>>) -> !fir.ref<!fir.array<100xi32>>
  %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
  fir.store %v to %e : !fir.ref<i32>
  fir.freemem %0 : !fir.heap<!fir.array<100xi32>>
  return
}

}
