
// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QPtest_coarray_get_scalar() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QFtest_coarray_get_scalarEa) : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
  %2:2 = hlfir.declare %1 {uniq_name = "_QFtest_coarray_get_scalarEa"} : (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>)
  %3 = fir.alloca f32 {bindc_name = "b", uniq_name = "_QFtest_coarray_get_scalarEb"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFtest_coarray_get_scalarEb"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %cst = arith.constant 2.000000e+00 : f32
  hlfir.assign %cst to %4#0 : f32, !fir.ref<f32>
  %5 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
  %6 = fir.box_addr %5 : (!fir.box<!fir.heap<f32>, corank:1>) -> !fir.heap<f32>
  %7 = hlfir.designate %6 : (!fir.heap<f32>) -> !fir.ref<f32>
  %c2_i64 = arith.constant 2 : i64
  mif.get_coarray from %7[%c2_i64] to %4#0 : (!fir.ref<f32>, i64, !fir.ref<f32>) -> ()
  return
}

// CHECK: %[[VAL_0:.*]] = fir.alloca !fir.box<f32>
// CHECK: %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
// CHECK: %[[VAL_2:.*]] = fir.alloca i32
// CHECK: %[[VAL_3:.*]] = fir.alloca i64
// CHECK: %[[VAL_4:.*]] = fir.alloca i64
// CHECK: %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK: %[[VAL_6:.*]] = fir.address_of(@_QFtest_coarray_get_scalarEa) : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
// CHECK: %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFtest_coarray_get_scalarEa"} : (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>)
// CHECK: %[[VAL_8:.*]] = fir.alloca f32 {bindc_name = "b", uniq_name = "_QFtest_coarray_get_scalarEb"}
// CHECK: %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFtest_coarray_get_scalarEb"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
// CHECK: %cst = arith.constant 2.000000e+00 : f32
// CHECK: hlfir.assign %cst to %[[VAL_9]]#0 : f32, !fir.ref<f32>
// CHECK: %[[VAL_10:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
// CHECK: %[[VAL_11:.*]] = fir.box_addr %[[VAL_10]] : (!fir.box<!fir.heap<f32>, corank:1>) -> !fir.heap<f32>
// CHECK: %[[VAL_12:.*]] = hlfir.designate %[[VAL_11]]   : (!fir.heap<f32>) -> !fir.ref<f32>
// CHECK: %c2_i64 = arith.constant 2 : i64
// CHECK: %c4_i64 = arith.constant 4 : i64
// CHECK: fir.store %c4_i64 to %[[VAL_4]] : !fir.ref<i64>
// CHECK: %[[VAL_13:.*]] = fir.address_of(@_QFtest_coarray_get_scalarEa_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>
// CHECK: %[[VAL_14:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_15:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %c0_i64 = arith.constant 0 : i64
// CHECK: fir.store %c0_i64 to %[[VAL_3]] : !fir.ref<i64>
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %[[VAL_16:.*]] = fir.coordinate_of %[[VAL_1]], %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CHECK: fir.store %c2_i64 to %[[VAL_16]] : !fir.ref<i64>
// CHECK: %[[VAL_17:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CHECK: %[[VAL_18:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_19:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CHECK: %[[VAL_20:.*]] = fir.convert %[[VAL_17]] : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK: fir.call @_QMprifPprif_initial_team_index(%[[VAL_19]], %[[VAL_20]], %[[VAL_2]], %[[VAL_18]]) : (!fir.ref<none>, !fir.box<!fir.array<?xi64>>, !fir.ref<i32>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_21:.*]] = fir.embox %[[VAL_9]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
// CHECK: fir.store %[[VAL_21]] to %[[VAL_0]] : !fir.ref<!fir.box<f32>>
// CHECK: %[[VAL_22:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CHECK: %[[VAL_23:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<f32>>) -> !fir.ptr<none>
// CHECK: fir.call @_QMprifPprif_get(%[[VAL_2]], %[[VAL_22]], %[[VAL_3]], %[[VAL_23]], %[[VAL_4]], %[[VAL_14]], %[[VAL_15]], %[[VAL_15]]) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i64>, !fir.ptr<none>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

