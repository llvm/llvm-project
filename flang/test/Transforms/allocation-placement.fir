// Test the default (-fno-stack-arrays) allocation-placement policy:
//  - small constant-size arrays go on the stack (within the budget),
//  - big constant-size arrays: user variables stay on the stack, temporaries
//    go on the heap,
//  - runtime-sized arrays go on the heap.
// A user variable is identified by a non-empty uniq_name; a temporary has none.
// i32 is 4 bytes, so <10xi32> = 40 bytes (small) and <100xi32> = 400 bytes (big)
// with the default 64-byte small threshold.

// RUN: fir-opt --allocation-placement %s | FileCheck %s

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {

// Small temporary heap allocation -> stack.
// CHECK-LABEL: func.func @small_temp
// CHECK: fir.alloca !fir.array<10xi32>
// CHECK-NOT: fir.allocmem
func.func @small_temp() {
  %0 = fir.allocmem !fir.array<10xi32>
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  %r = fir.convert %0 : (!fir.heap<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  fir.store %v to %e : !fir.ref<i32>
  fir.freemem %0 : !fir.heap<!fir.array<10xi32>>
  return
}

// Small temporary stack allocation -> stays on the stack.
// CHECK-LABEL: func.func @small_temp_alloca
// CHECK: fir.alloca !fir.array<10xi32>
// CHECK-NOT: fir.allocmem
func.func @small_temp_alloca() {
  %0 = fir.alloca !fir.array<10xi32>
  return
}

// Big temporary heap allocation -> stays on the heap.
// CHECK-LABEL: func.func @big_temp
// CHECK: fir.allocmem !fir.array<100xi32>
// CHECK: fir.freemem
func.func @big_temp() {
  %0 = fir.allocmem !fir.array<100xi32>
  fir.freemem %0 : !fir.heap<!fir.array<100xi32>>
  return
}

// Big user-variable stack allocation -> stays on the stack.
// CHECK-LABEL: func.func @big_user
// CHECK: fir.alloca !fir.array<100xi32>
// CHECK-NOT: fir.allocmem
func.func @big_user() {
  %0 = fir.alloca !fir.array<100xi32> {bindc_name = "arr", uniq_name = "_QFbig_userEarr"}
  return
}

// Big temporary stack allocation -> heap.
// CHECK-LABEL: func.func @big_temp_alloca
// CHECK: fir.allocmem !fir.array<100xi32>
// CHECK: fir.freemem
func.func @big_temp_alloca() {
  %0 = fir.alloca !fir.array<100xi32>
  return
}

// Runtime-sized user variable (automatic array) -> heap.
// CHECK-LABEL: func.func @dyn_user
// CHECK: fir.allocmem !fir.array<?xi32>
// CHECK: fir.freemem
func.func @dyn_user(%n: index) {
  %0 = fir.alloca !fir.array<?xi32>, %n {bindc_name = "arr", uniq_name = "_QFdyn_userEarr"}
  return
}

// Runtime-sized temporary -> stays on the heap.
// CHECK-LABEL: func.func @dyn_temp
// CHECK: fir.allocmem !fir.array<?xi32>
func.func @dyn_temp(%n: index) {
  %0 = fir.allocmem !fir.array<?xi32>, %n
  fir.freemem %0 : !fir.heap<!fir.array<?xi32>>
  return
}

}
