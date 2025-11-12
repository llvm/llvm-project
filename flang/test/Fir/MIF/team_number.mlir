// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QQmain() attributes {fir.bindc_name = "TEST_TEAM_NUMBER"} {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "t", uniq_name = "_QFEt"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFEt"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}> {bindc_name = "team", uniq_name = "_QFEteam"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFEteam"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
  %5 = fir.address_of(@_QQ_QM__fortran_builtinsT__builtin_team_type.DerivedInit) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
  fir.copy %5 to %4#0 no_overlap : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
  %6 = fir.embox %4#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
  %7 = mif.team_number team %6 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> i64
  %8 = fir.convert %7 : (i64) -> i32
  hlfir.assign %8 to %2#0 : i32, !fir.ref<i32>
  %9 = mif.team_number : () -> i64
  %10 = fir.convert %9 : (i64) -> i32
  hlfir.assign %10 to %2#0 : i32, !fir.ref<i32>
  return
}

// CHECK: %[[VAL_1:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_team_number(%[[VAL_1]], %[[RESULT:.*]]) : (!fir.box<none>, !fir.ref<i64>) -> ()
// CHECK: %[[VAL_2:.*]] = fir.load %[[RESULT]] : !fir.ref<i64>

// CHECK: %[[VAL_3:.*]] = fir.absent !fir.box<none>
// CHECK: fir.call @_QMprifPprif_team_number(%[[VAL_3]], %[[RESULT:.*]]) : (!fir.box<none>, !fir.ref<i64>) -> ()
// CHECK: %[[VAL_4:.*]] = fir.load %[[RESULT]] : !fir.ref<i64>
