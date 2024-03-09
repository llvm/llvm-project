// RUN: fir-opt --memory-allocation-opt="dynamic-array-on-heap=true maximum-array-alloc-size=1024" %s | FileCheck %s

// Test for size of array being too big.

// CHECK-LABEL: func @_QPs1(
// CHECK: %[[mem:.*]] = fir.allocmem !fir.array<1000123xi32> {bindc_name = "array", uniq_name = "_QFs1Earray"}
// CHECK: fir.call @_QPs3(
// CHECK: fir.freemem %[[mem]]
// CHECK-NEXT: return

func.func @_QPs1() {
  %0 = fir.alloca !fir.array<1000123xi32> {bindc_name = "array", uniq_name = "_QFs1Earray"}
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

