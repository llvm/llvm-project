// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QQmain() attributes {fir.bindc_name = "TEST_CHANGE_TEAM"} {
  %0 = fir.dummy_scope : !fir.dscope
  %c10 = arith.constant 10 : index
  %1 = fir.alloca !fir.char<1,10> {bindc_name = "err", uniq_name = "_QFEerr"}
  %2:2 = hlfir.declare %1 typeparams %c10 {uniq_name = "_QFEerr"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca i32 {bindc_name = "stat", uniq_name = "_QFEstat"}
  %6:2 = hlfir.declare %5 {uniq_name = "_QFEstat"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}> {bindc_name = "team", uniq_name = "_QFEteam"}
  %8:2 = hlfir.declare %7 {uniq_name = "_QFEteam"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
  %9 = fir.address_of(@_QQ_QM__fortran_builtinsT__builtin_team_type.DerivedInit) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
  fir.copy %9 to %8#0 no_overlap : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
  %10 = fir.embox %8#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
  mif.change_team %10 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) {
    %13 = fir.load %4#0 : !fir.ref<i32>
    %c1_i32 = arith.constant 1 : i32
    %14 = arith.addi %13, %c1_i32 : i32
    hlfir.assign %14 to %4#0 : i32, !fir.ref<i32>
    mif.end_team : () -> ()
  }
  %11 = fir.embox %2#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
  %12 = fir.embox %8#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
  mif.change_team %12 stat %6#0 errmsg %11 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<i32>, !fir.box<!fir.char<1,10>>) {
    mif.end_team : () -> ()
  }
  return
}

// CHECK: %[[VAL_1:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_2:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_3:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_change_team(%[[VAL_3]], %[[VAL_1]], %[[VAL_2]], %[[VAL_2]]) : (!fir.box<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[VAL_4:.*]] = fir.load %[[VAR_1:.*]]#0 : !fir.ref<i32>
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[VAL_5:.*]] = arith.addi %[[VAL_4]], %[[C1]] : i32
// CHECK: hlfir.assign %[[VAL_5]] to %[[VAR_1]]#0 : i32, !fir.ref<i32>
// CHECK: %[[VAL_6:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_7:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_end_team(%[[VAL_6]], %[[VAL_7]], %[[VAL_7]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_8:.*]] = fir.embox %[[ERRMSG:.*]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>
// CHECK: %[[VAL_9:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[TEAM_2:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_change_team(%[[TEAM_2]], %[[STAT:.*]]#0, %[[VAL_10]], %[[VAL_9]]) : (!fir.box<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[VAL_11:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_12:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_end_team(%[[VAL_11]], %[[VAL_12]], %[[VAL_12]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
