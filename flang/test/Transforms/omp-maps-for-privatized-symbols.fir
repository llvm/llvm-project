// RUN: fir-opt --split-input-file --omp-maps-for-privatized-symbols %s | FileCheck %s
module attributes {omp.is_target_device = false} {
  omp.private {type = private} @_QFtarget_simpleEsimple_var_private_ref_box_heap_i32 : !fir.ref<!fir.box<!fir.heap<i32>>> alloc {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>>):
    %0 = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "simple_var", pinned, uniq_name = "_QFtarget_simpleEsimple_var"}
    %1 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %5:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_simpleEsimple_var"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
    omp.yield(%5#0 : !fir.ref<!fir.box<!fir.heap<i32>>>)
  }
  func.func @_QPtarget_simple() {
    %0 = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFtarget_simpleEa"}
    %1:2 = hlfir.declare %0 {uniq_name = "_QFtarget_simpleEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "simple_var", uniq_name = "_QFtarget_simpleEsimple_var"}
    %3 = fir.zero_bits !fir.heap<i32>
    %4 = fir.embox %3 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
    fir.store %4 to %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %5:2 = hlfir.declare %2 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_simpleEsimple_var"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
    %c2_i32 = arith.constant 2 : i32
    hlfir.assign %c2_i32 to %1#0 : i32, !fir.ref<i32>
    %6 = omp.map.info var_ptr(%1#1 : !fir.ref<i32>, i32) map_clauses(to) capture(ByRef) -> !fir.ref<i32> {name = "a"}
    omp.target map_entries(%6 -> %arg0 : !fir.ref<i32>) private(@_QFtarget_simpleEsimple_var_private_ref_box_heap_i32 %5#0 -> %arg1 : !fir.ref<!fir.box<!fir.heap<i32>>>) {
      %11:2 = hlfir.declare %arg0 {uniq_name = "_QFtarget_simpleEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %12:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_simpleEsimple_var"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
      %c10_i32 = arith.constant 10 : i32
      %13 = fir.load %11#0 : !fir.ref<i32>
      %14 = arith.addi %c10_i32, %13 : i32
      hlfir.assign %14 to %12#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
      omp.terminator
    }
    %7 = fir.load %5#1 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %8 = fir.box_addr %7 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
    %9 = fir.convert %8 : (!fir.heap<i32>) -> i64
    %c0_i64 = arith.constant 0 : i64
    %10 = arith.cmpi ne, %9, %c0_i64 : i64
    fir.if %10 {
      %11 = fir.load %5#1 : !fir.ref<!fir.box<!fir.heap<i32>>>
      %12 = fir.box_addr %11 : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
      fir.freemem %12 : !fir.heap<i32>
      %13 = fir.zero_bits !fir.heap<i32>
      %14 = fir.embox %13 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
      fir.store %14 to %5#1 : !fir.ref<!fir.box<!fir.heap<i32>>>
    }
    return
  }
}
// CHECK: %[[MAP0:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<i32>, i32) map_clauses(to) capture(ByRef) -> !fir.ref<i32> {name = "a"}
// CHECK: %[[MAP1:.*]] =  omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<i32>>>    
// CHECK: omp.target map_entries(%[[MAP0]] -> %arg0, %[[MAP1]] -> %arg1 : !fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<i32>>>)
