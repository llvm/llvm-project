// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// CHECK-LABEL:   func.func @_QPtarget_simple() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFtarget_simpleEa"}
// CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFtarget_simpleEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "simple_var", uniq_name = "_QFtarget_simpleEsimple_var"}
// CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.heap<i32>
// CHECK:           %[[VAL_5:.*]] = fir.embox %[[VAL_4]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
// CHECK:           fir.store %[[VAL_5]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_3]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_simpleEsimple_var"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
// CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>
// CHECK:           %[[VAL_7:.*]] = omp.map.info var_ptr(%[[VAL_2]]#1 : !fir.ref<i32>, i32) map_clauses(to) capture(ByRef) -> !fir.ref<i32> {name = "a"}
// CHECK:           omp.target map_entries(%[[VAL_7]] -> %[[VAL_8:.*]] : !fir.ref<i32>) private(@_QFtarget_simpleEsimple_var_private_ref_box_heap_i32 %[[VAL_6]]#0 -> %[[VAL_9:.*]] : !fir.ref<!fir.box<!fir.heap<i32>>>) {
// CHECK:             %[[VAL_10:.*]] = arith.constant 10 : i32
// CHECK:             %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFtarget_simpleEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:             %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_9]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_simpleEsimple_var"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
// CHECK:             %[[VAL_13:.*]] = fir.load %[[VAL_11]]#0 : !fir.ref<i32>
// CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_10]] : i32
// CHECK:             hlfir.assign %[[VAL_14]] to %[[VAL_12]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
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
    omp.target map_entries(%6 -> %arg0 : !fir.ref<i32>) private(@_QFtarget_simpleEsimple_var_private_ref_box_heap_i32 %5#0 -> %arg1 : !fir.ref<!fir.box<!fir.heap<i32>>>){
        omp.teams {
            omp.workdistribute {
                %11:2 = hlfir.declare %arg0 {uniq_name = "_QFtarget_simpleEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
                %12:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_simpleEsimple_var"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
                %c10_i32 = arith.constant 10 : i32
                %13 = fir.load %11#0 : !fir.ref<i32>
                %14 = arith.addi %c10_i32, %13 : i32
                hlfir.assign %14 to %12#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
                omp.terminator
            }
            omp.terminator
        }
        omp.terminator
    }
    return
}
