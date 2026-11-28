// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QQmain() attributes {fir.bindc_name = "IMAGE_STATUS_TEST"} {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i64 <{bindc_name = "d", uniq_name = "_QFEd"}>
  %2:2 = hlfir.declare %1 uniq_name("_QFEd") : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %3 = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFEi"}>
  %4:2 = hlfir.declare %3 uniq_name("_QFEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca i32 <{bindc_name = "s", uniq_name = "_QFEs"}>
  %6:2 = hlfir.declare %5 uniq_name("_QFEs") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.address_of(@_QFEt) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>
  %8:2 = hlfir.declare %7 uniq_name("_QFEt") : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>)
  %9 = fir.load %4#0 : !fir.ref<i32>
  %10 = mif.image_status %9 : (i32) -> i32
  hlfir.assign %10 to %6#0 : i32, !fir.ref<i32>
  %c1_i32 = arith.constant 1 : i32
  %11 = mif.image_status %c1_i32 : (i32) -> i32
  hlfir.assign %11 to %6#0 : i32, !fir.ref<i32>
  %12 = fir.load %4#0 : !fir.ref<i32>
  %13 = mif.image_status %12 : (i32) -> i32
  %14 = fir.convert %13 : (i32) -> i64
  hlfir.assign %14 to %2#0 : i64, !fir.ref<i64>
  %15 = fir.load %4#0 : !fir.ref<i32>
  %16 = mif.image_status %15 team %8#0 : (i32, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>) -> i32
  hlfir.assign %16 to %6#0 : i32, !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @_QQmain()
// CHECK: %[[VAL_0:.*]] = fir.alloca i32
// CHECK: %[[VAL_1:.*]] = fir.alloca i32
// CHECK: %[[VAL_2:.*]] = fir.alloca i32
// CHECK: %[[VAL_3:.*]] = fir.alloca i32
// CHECK: %[[VAL_4:.*]] = fir.alloca i32
// CHECK: %[[VAL_5:.*]] = fir.alloca i32
// CHECK: %[[VAL_6:.*]] = fir.alloca i32
// CHECK: %[[VAL_7:.*]] = fir.alloca i32
// CHECK: %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK: %[[VAL_9:.*]] = fir.alloca i64 <{bindc_name = "d", uniq_name = "_QFEd"}>
// CHECK: %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] uniq_name("_QFEd") : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
// CHECK: %[[VAL_11:.*]] = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFEi"}>
// CHECK: %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] uniq_name("_QFEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_13:.*]] = fir.alloca i32 <{bindc_name = "s", uniq_name = "_QFEs"}>
// CHECK: %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]] uniq_name("_QFEs") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_15:.*]] = fir.address_of(@_QFEt) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{{.*}}
// CHECK: %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_15]] uniq_name("_QFEt") : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{{.*}}

// CHECK: %[[VAL_17:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<i32>
// CHECK: fir.store %[[VAL_17]] to %[[VAL_6]] : !fir.ref<i32>
// CHECK: %[[VAL_18:.*]] = fir.absent !fir.ref<none>
// CHECK: fir.call @_QMprifPprif_image_status(%[[VAL_6]], %[[VAL_18]], %[[VAL_7]]) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_19:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK: hlfir.assign %[[VAL_19]] to %[[VAL_14]]#0 : i32, !fir.ref<i32>

// CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK: fir.store %[[C1_I32]] to %[[VAL_4]] : !fir.ref<i32>
// CHECK: %[[VAL_20:.*]] = fir.absent !fir.ref<none>
// CHECK: fir.call @_QMprifPprif_image_status(%[[VAL_4]], %[[VAL_20]], %[[VAL_5]]) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_21:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
// CHECK: hlfir.assign %[[VAL_21]] to %[[VAL_14]]#0 : i32, !fir.ref<i32>

// CHECK: %[[VAL_22:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<i32>
// CHECK: fir.store %[[VAL_22]] to %[[VAL_2]] : !fir.ref<i32>
// CHECK: %[[VAL_23:.*]] = fir.absent !fir.ref<none>
// CHECK: fir.call @_QMprifPprif_image_status(%[[VAL_2]], %[[VAL_23]], %[[VAL_3]]) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_24:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
// CHECK: %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
// CHECK: hlfir.assign %[[VAL_25]] to %[[VAL_10]]#0 : i64, !fir.ref<i64>

// CHECK: %[[VAL_26:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<i32>
// CHECK: fir.store %[[VAL_26]] to %[[VAL_0]] : !fir.ref<i32>
// CHECK: %[[VAL_27:.*]] = fir.convert %[[VAL_16]]#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{{.*}} -> !fir.ref<none>
// CHECK: fir.call @_QMprifPprif_image_status(%[[VAL_0]], %[[VAL_27]], %[[VAL_1]]) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_28:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
// CHECK: hlfir.assign %[[VAL_28]] to %[[VAL_14]]#0 : i32, !fir.ref<i32>
// CHECK: return
