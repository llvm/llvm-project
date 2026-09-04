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

// Small temporary inside an OpenACC compute construct -> stack, but the
// allocation must stay inside the construct.
// CHECK-LABEL: func.func @small_temp_in_acc_parallel
// CHECK-NOT: fir.alloca
// CHECK: acc.parallel {
// CHECK-NEXT: fir.alloca !fir.array<10xi32>
// CHECK-NOT: fir.allocmem
func.func @small_temp_in_acc_parallel() {
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  acc.parallel {
    %0 = fir.allocmem !fir.array<10xi32>
    %r = fir.convert %0 : (!fir.heap<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
    %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
    fir.store %v to %e : !fir.ref<i32>
    fir.freemem %0 : !fir.heap<!fir.array<10xi32>>
    acc.yield
  }
  return
}

// Tail-padded record array: !fir.array<3x!fir.type<tpad{a:i32,b:i8}>>
// Each element is 8 bytes (5 store + 3 tail padding), total = 24 bytes.
// SequenceType rounds each element to its alignment boundary (alignTo(5,4)=8)
// before multiplying by the element count, so both the pre-fix and post-fix
// code produce 24 bytes.  This case verifies that per-element SequenceType
// rounding is correct, not that the tail-padding fix changes the total.
// With the default 64-byte threshold the 24-byte array is promoted to stack.
// CHECK-LABEL: func.func @tail_padded_record_placement
// CHECK: fir.alloca !fir.array<3x!fir.type<tpad{a:i32,b:i8}>>
// CHECK-NOT: fir.allocmem
func.func @tail_padded_record_placement() {
  %0 = fir.allocmem !fir.array<3x!fir.type<tpad{a:i32,b:i8}>>
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  %r = fir.convert %0 : (!fir.heap<!fir.array<3x!fir.type<tpad{a:i32,b:i8}>>>) -> !fir.ref<!fir.array<3x!fir.type<tpad{a:i32,b:i8}>>>
  %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<3x!fir.type<tpad{a:i32,b:i8}>>>, index) -> !fir.ref<!fir.type<tpad{a:i32,b:i8}>>
  %f = fir.coordinate_of %e, %c0 : (!fir.ref<!fir.type<tpad{a:i32,b:i8}>>, index) -> !fir.ref<i32>
  fir.store %v to %f : !fir.ref<i32>
  fir.freemem %0 : !fir.heap<!fir.array<3x!fir.type<tpad{a:i32,b:i8}>>>
  return
}

// Packed record array: !fir.array<5x!fir.type<tpk<{i:i32,d:f64}>>>
// Each element is 12 bytes (packed: 4+8, no gaps or tail padding), total=60B.
// Without the isPacked fix, getTypeSizeAndAlignment returns 16 per element
// (aligned like an unpacked struct), giving a wrong total of 80 bytes which
// exceeds the 64-byte threshold and would select heap placement.  With the
// fix the correct total of 60 bytes is below the threshold and the array
// must be promoted to stack.
// CHECK-LABEL: func.func @packed_record_placement
// CHECK: fir.alloca !fir.array<5x!fir.type<tpk<{i:i32,d:f64}>>>
// CHECK-NOT: fir.allocmem
func.func @packed_record_placement() {
  %0 = fir.allocmem !fir.array<5x!fir.type<tpk<{i:i32,d:f64}>>>
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  %r = fir.convert %0 : (!fir.heap<!fir.array<5x!fir.type<tpk<{i:i32,d:f64}>>>>) -> !fir.ref<!fir.array<5x!fir.type<tpk<{i:i32,d:f64}>>>>
  %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<5x!fir.type<tpk<{i:i32,d:f64}>>>>, index) -> !fir.ref<!fir.type<tpk<{i:i32,d:f64}>>>
  %f = fir.coordinate_of %e, %c0 : (!fir.ref<!fir.type<tpk<{i:i32,d:f64}>>>, index) -> !fir.ref<i32>
  fir.store %v to %f : !fir.ref<i32>
  fir.freemem %0 : !fir.heap<!fir.array<5x!fir.type<tpk<{i:i32,d:f64}>>>>
  return
}

}
