// RUN: fir-opt --split-input-file --omp-maps-for-privatized-symbols %s | FileCheck %s
module attributes {omp.is_target_device = false} {
  omp.private {type = private} @_QFtarget_simpleEsimple_var_private_ref_box_heap_i32 : !fir.box<!fir.heap<i32>> init {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<i32>>>, %arg1: !fir.ref<!fir.box<!fir.heap<i32>>>):
    %mold = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<i32>>>
    // extract box address, see if it is null, etc
    omp.yield(%arg1: !fir.ref<!fir.box<!fir.heap<i32>>>)
  }
  omp.private {type = firstprivate} @_QFtarget_simpleEfp_int_firstprivate_i32 : i32 copy {
  ^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
    %0 = fir.load %arg0 : !fir.ref<i32>
    hlfir.assign %0 to %arg1 : i32, !fir.ref<i32>
    omp.yield(%arg1 : !fir.ref<i32>)
  }
  func.func @_QPtarget_simple() {
    %0 = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFtarget_simpleEa"}
    %1:2 = hlfir.declare %0 {uniq_name = "_QFtarget_simpleEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "simple_var", uniq_name = "_QFtarget_simpleEsimple_var"}
    %3 = fir.zero_bits !fir.heap<i32>
    %4 = fir.embox %3 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
    fir.store %4 to %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %5:2 = hlfir.declare %2 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_simpleEsimple_var"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
    %6 = fir.alloca i32 {bindc_name = "fp_int", uniq_name = "_QFtarget_simpleEfp_int"}
    %7:2 = hlfir.declare %6 {uniq_name = "_QFtarget_simpleEfp_int"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c2_i32 = arith.constant 2 : i32
    hlfir.assign %c2_i32 to %1#0 : i32, !fir.ref<i32>
    %8 = omp.map.info var_ptr(%1#1 : !fir.ref<i32>, i32) map_clauses(to) capture(ByRef) -> !fir.ref<i32> {name = "a"}
    omp.target map_entries(%8 -> %arg0 : !fir.ref<i32>) private(@_QFtarget_simpleEsimple_var_private_ref_box_heap_i32 %5#0 -> %arg1, @_QFtarget_simpleEfp_int_firstprivate_i32 %7#0 -> %arg2  : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<i32>) {
      omp.terminator
    }
    return
  }
}
// CHECK: %[[MAP0:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<i32>, i32) map_clauses(to) capture(ByRef) -> !fir.ref<i32> {name = "a"}
// CHECK: %[[MAP1:.*]] =  omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK: %[[MAP2:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<i32>, i32) map_clauses(to) capture(ByCopy) -> !fir.ref<i32>
// CHECK:  omp.target map_entries(%[[MAP0]] -> %arg0, %[[MAP1]] -> %arg1, %[[MAP2]] -> %arg2 : !fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<i32>)
