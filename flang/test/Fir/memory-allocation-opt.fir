// The default (-fno-stack-arrays) allocation-placement policy sends big
// constant-size temporaries and runtime-sized arrays to the heap, reproducing
// the legacy memory-allocation-opt size-based heap placement. A data layout is
// required to compute the constant byte size of _QPs1.
// RUN: fir-opt --allocation-placement %s | FileCheck %s

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {

// Test for size of array being too big: a big constant-size temporary goes on
// the heap.

// CHECK-LABEL: func @_QPs1(
// CHECK: %[[mem:.*]] = fir.allocmem !fir.array<1000123xi32> {bindc_name = "", uniq_name = ""}
// CHECK: fir.call @_QPs3(
// CHECK: fir.freemem %[[mem]]
// CHECK-NEXT: return

func.func @_QPs1() {
  %0 = fir.alloca !fir.array<1000123xi32>
  fir.call @_QPs3(%0) : (!fir.ref<!fir.array<1000123xi32>>) -> ()
  return
}

// Test for dynamic array.

// CHECK-LABEL: func @_QPs2(
// CHECK: %[[mem:.*]] = fir.allocmem !fir.array<?xi32>, %{{[0-9]+}} {bindc_name = "array", uniq_name = "_QFs2Earray"}
// CHECK: fir.call @_QPs3(
// CHECK: fir.freemem %[[mem]]
// CHECK-NEXT: return

func.func @_QPs2(%arg0: !fir.ref<i32>) {
  %0 = fir.load %arg0 : !fir.ref<i32>
  %1 = fir.convert %0 : (i32) -> index
  %2 = fir.alloca !fir.array<?xi32>, %1 {bindc_name = "array", uniq_name = "_QFs2Earray"}
  %3 = fir.convert %2 : (!fir.ref<!fir.array<?xi32>>) -> !fir.ref<!fir.array<1000123xi32>>
  fir.call @_QPs3(%3) : (!fir.ref<!fir.array<1000123xi32>>) -> ()
  return
}
func.func private @_QPs3(!fir.ref<!fir.array<1000123xi32>>)
}
