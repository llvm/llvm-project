// Test the per-function stack budget: small arrays are placed on the stack
// until the total stack limit is reached, after which further small arrays are
// left on the heap. With a 64-byte limit, the first 40-byte array fits on the
// stack but the following 48-byte array does not (40 + 48 = 88 > 64).

// RUN: fir-opt --allocation-placement="small-array-threshold=64 total-stack-limit=64" %s | FileCheck %s

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {

// CHECK-LABEL: func.func @budget
// CHECK-DAG: fir.alloca !fir.array<10xi32>
// CHECK-DAG: fir.allocmem !fir.array<12xi32>
func.func @budget() {
  %0 = fir.allocmem !fir.array<10xi32>
  %1 = fir.allocmem !fir.array<12xi32>
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  %r0 = fir.convert %0 : (!fir.heap<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %e0 = fir.coordinate_of %r0, %c0 : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  fir.store %v to %e0 : !fir.ref<i32>
  %r1 = fir.convert %1 : (!fir.heap<!fir.array<12xi32>>) -> !fir.ref<!fir.array<12xi32>>
  %e1 = fir.coordinate_of %r1, %c0 : (!fir.ref<!fir.array<12xi32>>, index) -> !fir.ref<i32>
  fir.store %v to %e1 : !fir.ref<i32>
  fir.freemem %0 : !fir.heap<!fir.array<10xi32>>
  fir.freemem %1 : !fir.heap<!fir.array<12xi32>>
  return
}

// A requested alignment counts against the budget as worst-case padding
// (alignment - 1), both when deciding and when accounting: the 40-byte array
// is below the small-array threshold, but 40 + 63 bytes of padding do not fit
// the 64-byte limit, so it stays on the heap.
// CHECK-LABEL: func.func @budget_aligned_too_big
// CHECK: fir.allocmem !fir.array<10xi32> <{alignment = 64 : i64
// CHECK-NOT: fir.alloca
func.func @budget_aligned_too_big() {
  %0 = fir.allocmem !fir.array<10xi32> {alignment = 64 : i64}
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  %r0 = fir.convert %0 : (!fir.heap<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %e0 = fir.coordinate_of %r0, %c0 : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  fir.store %v to %e0 : !fir.ref<i32>
  fir.freemem %0 : !fir.heap<!fir.array<10xi32>>
  return
}

// An 8-byte array aligned to 32 fits (8 + 31 = 39 <= 64) and goes on the
// stack; its padding is then charged, so the following 40-byte array no longer
// fits (39 + 40 = 79 > 64) and stays on the heap.
// CHECK-LABEL: func.func @budget_aligned_padding_charged
// CHECK-DAG: fir.alloca !fir.array<2xi32> <{alignment = 32 : i64}>
// CHECK-DAG: fir.allocmem !fir.array<10xi32>
func.func @budget_aligned_padding_charged() {
  %0 = fir.allocmem !fir.array<2xi32> {alignment = 32 : i64}
  %1 = fir.allocmem !fir.array<10xi32>
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  %r0 = fir.convert %0 : (!fir.heap<!fir.array<2xi32>>) -> !fir.ref<!fir.array<2xi32>>
  %e0 = fir.coordinate_of %r0, %c0 : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
  fir.store %v to %e0 : !fir.ref<i32>
  %r1 = fir.convert %1 : (!fir.heap<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %e1 = fir.coordinate_of %r1, %c0 : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  fir.store %v to %e1 : !fir.ref<i32>
  fir.freemem %0 : !fir.heap<!fir.array<2xi32>>
  fir.freemem %1 : !fir.heap<!fir.array<10xi32>>
  return
}

}
