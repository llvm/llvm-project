// RUN: fir-opt --mif-convert %s | FileCheck %s

  func.func @_QQmain() attributes {fir.bindc_name = "TEST_TEAM_NUMBER"} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "t", uniq_name = "_QFEt"}
    %2:2 = hlfir.declare %1 {uniq_name = "_QFEt"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.address_of(@_QFEteam) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>
    %4:2 = hlfir.declare %3 {uniq_name = "_QFEteam"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>)
    %5 = mif.team_number team %4#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>) -> i64
    %6 = fir.convert %5 : (i64) -> i32
    hlfir.assign %6 to %2#0 : i32, !fir.ref<i32>
    %7 = mif.team_number : () -> i64
    %8 = fir.convert %7 : (i64) -> i32
    hlfir.assign %8 to %2#0 : i32, !fir.ref<i32>
    return
  }

// CHECK: %[[VAL_1:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.ref<none>
// CHECK: fir.call @_QMprifPprif_team_number(%[[VAL_1]], %[[RESULT:.*]]) : (!fir.ref<none>, !fir.ref<i64>) -> ()
// CHECK: %[[VAL_2:.*]] = fir.load %[[RESULT]] : !fir.ref<i64>

// CHECK: %[[VAL_3:.*]] = fir.absent !fir.ref<none>
// CHECK: fir.call @_QMprifPprif_team_number(%[[VAL_3]], %[[RESULT:.*]]) : (!fir.ref<none>, !fir.ref<i64>) -> ()
// CHECK: %[[VAL_4:.*]] = fir.load %[[RESULT]] : !fir.ref<i64>
