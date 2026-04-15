// RUN: fir-opt -flang-licm --split-input-file %s | FileCheck %s

// Test that a load of a scalar allocated by fir.allocmem
// is hoisted out of the loop (the allocation proves
// the variable is always present).
// CHECK-LABEL:   func.func @test_allocmem(
// CHECK:           %[[ALLOCMEM:.*]] = fir.allocmem f32
// CHECK:           %[[DECLARE:.*]] = fir.declare %[[ALLOCMEM]] {uniq_name = "_QFtestEy"}
// CHECK:           %[[LOAD:.*]] = fir.load %[[DECLARE]]
// CHECK:           fir.do_loop
// CHECK-NOT:         fir.load
func.func @test_allocmem(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.allocmem f32 {uniq_name = "_QFtestEy.alloc"}
  %2 = fir.declare %1 {uniq_name = "_QFtestEy"} : (!fir.heap<f32>) -> !fir.heap<f32>
  %3 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtestEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.load %3 : !fir.ref<i32>
  %n = fir.convert %4 : (i32) -> index
  %5 = fir.shape %n : (index) -> !fir.shape<1>
  %6 = fir.declare %arg0(%5) dummy_scope %0 arg 1 {uniq_name = "_QFtestEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xf32>>
  fir.do_loop %arg2 = %c1 to %n step %c1 {
    %7 = fir.load %2 : !fir.heap<f32>
    %8 = fir.array_coor %6(%5) %arg2 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    fir.store %7 to %8 : !fir.ref<f32>
  }
  fir.freemem %2 : !fir.heap<f32>
  return
}

// -----

// Test that a load of a scalar allocated by cuf.alloc
// is hoisted out of the loop (the allocation proves
// the variable is always present).
// CHECK-LABEL:   func.func @test_cuf_alloc(
// CHECK:           %[[ALLOC:.*]] = cuf.alloc f32
// CHECK:           %[[DECLARE:.*]] = fir.declare %[[ALLOC]]
// CHECK:           %[[LOAD:.*]] = fir.load %[[DECLARE]]
// CHECK:           fir.do_loop
// CHECK-NOT:         fir.load
func.func @test_cuf_alloc(%arg0: !fir.ref<!fir.array<?xf32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = cuf.alloc f32 {data_attr = #cuf.cuda<device>, uniq_name = "_QFtestEy"} -> !fir.ref<f32>
  %2 = fir.declare %1 {data_attr = #cuf.cuda<device>, uniq_name = "_QFtestEy"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %3 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtestEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.load %3 : !fir.ref<i32>
  %n = fir.convert %4 : (i32) -> index
  %5 = fir.shape %n : (index) -> !fir.shape<1>
  %6 = fir.declare %arg0(%5) dummy_scope %0 arg 1 {data_attr = #cuf.cuda<device>, uniq_name = "_QFtestEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xf32>>
  fir.do_loop %arg2 = %c1 to %n step %c1 {
    %7 = fir.load %2 : !fir.ref<f32>
    %8 = fir.array_coor %6(%5) %arg2 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    fir.store %7 to %8 : !fir.ref<f32>
  }
  cuf.free %1 : !fir.ref<f32> {data_attr = #cuf.cuda<device>}
  return
}
