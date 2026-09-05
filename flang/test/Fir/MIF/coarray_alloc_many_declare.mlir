// RUN: fir-opt --mif-convert %s | FileCheck %s

// This test verifies that, during the MIF conversion, when there are multiple `fir.declare` 
// operations for a same coarray_handle, the conversion continues without adding a second block to 
// created fir.global.

func.func @_QQmain() attributes {fir.bindc_name = "P"} {
  %0 = fir.alloca !fir.array<0xi64>
  %1 = fir.alloca !fir.array<1xi64>
  %2 = fir.dummy_scope : !fir.dscope
  %3 = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
  %4:2 = hlfir.declare %3 {fortran_attrs = #fir.var_attrs<allocatable, internal_assoc>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>)
  %5 = fir.absent !fir.box<none>
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %6 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
  fir.store %c1_i64 to %6 : !fir.ref<i64>
  %7 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
  %8 = fir.embox %0 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
  mif.alloc_coarray %4#0 lcobounds %7 ucobounds %8 errmsg %5 {uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.box<none>) -> ()
  fir.call @_QFPinner() fastmath<contract> : () -> ()
  return
}
func.func private @_QFPinner() attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
  %2:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>)
  %c1_i32 = arith.constant 1 : i32
  hlfir.assign %c1_i32 to %2#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
  return
}
fir.global internal @_QFEa : !fir.box<!fir.heap<i32>, corank:1> {
  %0 = fir.zero_bits !fir.heap<i32>
  %1 = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>, corank:1>
  fir.has_value %1 : !fir.box<!fir.heap<i32>, corank:1>
}

//CHECK-LABEL: @_QQmain
//CHECK-NEXT:  %[[VAL_0:.*]] = fir.alloca i64
//CHECK-NEXT:  %[[VAL_1:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>
//CHECK-NEXT:  %[[VAL_2:.*]] = fir.alloca !fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>
//CHECK-NEXT:  %[[VAL_3:.*]] = fir.alloca !fir.array<0xi64>
//CHECK-NEXT:  %[[VAL_4:.*]] = fir.alloca !fir.array<1xi64>
//CHECK-NEXT:  %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
//CHECK-NEXT:  %[[VAL_6:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
//CHECK-NEXT:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {fortran_attrs = #fir.var_attrs<allocatable, internal_assoc>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>)
//CHECK-NEXT:  %[[VAL_8:.*]] = fir.absent !fir.box<none>
//CHECK-NEXT:  %c1_i64 = arith.constant 1 : i64
//CHECK-NEXT:  %c0 = arith.constant 0 : index
//CHECK-NEXT:  %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_4]], %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
//CHECK-NEXT:  fir.store %c1_i64 to %[[VAL_9]] : !fir.ref<i64>
//CHECK-NEXT:  %[[VAL_10:.*]] = fir.embox %[[VAL_4]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
//CHECK-NEXT:  %[[VAL_11:.*]] = fir.embox %[[VAL_3]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
//CHECK-NEXT:  %[[VAL_12:.*]] = fir.zero_bits (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()
//CHECK-NEXT:  %[[VAL_13:.*]] = fir.emboxproc %[[VAL_12]] : ((!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()) -> !fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>
//CHECK-NEXT:  fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<!fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>>
//CHECK-NEXT:  %[[VAL_14:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>) -> !fir.ptr<none>
//CHECK-NEXT:  %[[VAL_15:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
//CHECK-NEXT:  %[[VAL_16:.*]] = fir.box_elesize %[[VAL_15]] : (!fir.box<!fir.heap<i32>, corank:1>) -> i64
//CHECK-NEXT:  fir.store %[[VAL_16]] to %[[VAL_0]] : !fir.ref<i64>
//CHECK-NEXT:  %[[VAL_17:.*]] = fir.absent !fir.ref<i32>
//CHECK-NEXT:  %[[VAL_18:.*]] = fir.absent !fir.box<!fir.char<1,?>>
//CHECK-NEXT:  %[[VAL_19:.*]] = fir.convert %[[VAL_10]] : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
//CHECK-NEXT:  %[[VAL_20:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.array<0xi64>>) -> !fir.box<!fir.array<?xi64>>
//CHECK-NEXT:  %[[VAL_21:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>>) -> !fir.ref<none>
//CHECK-NEXT:  %[[VAL_22:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
//CHECK-NEXT:  %[[VAL_23:.*]] = fir.convert %[[VAL_8]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
//CHECK-NEXT:  fir.call @_QMprifPprif_allocate_coarray(%[[VAL_19]], %[[VAL_20]], %[[VAL_0]], %[[VAL_21]], %[[VAL_22]], %[[VAL_14]], %[[VAL_17]], %[[VAL_23]], %[[VAL_18]]) : (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>, !fir.ref<i64>, !fir.ref<none>, !fir.ref<none>, !fir.ptr<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
//CHECK-NEXT:  %[[VAL_24:.*]] = fir.address_of(@_QFEa_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>
//CHECK-NEXT:  fir.copy %[[VAL_1]] to %[[VAL_24]] : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>
//CHECK-NEXT:  fir.call @_QFPinner() fastmath<contract> : () -> ()

//CHECK-LABEL: func.func private @_QFPinner() attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>}
//CHECK:       %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
//CHECK-NEXT:  %[[VAL_1:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
//CHECK-NEXT:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>)
//CHECK-NEXT:  %c1_i32 = arith.constant 1 : i32
//CHECK-NEXT:  hlfir.assign %c1_i32 to %[[VAL_2]]#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>

//CHECK-LABEL:  fir.global linkonce @_QFEa_coarray_handle : !fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>
//CHECK:    %[[VAL_0:.*]] = fir.zero_bits !fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>
//CHECK-NEXT:    fir.has_value %[[VAL_0]] : !fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>
