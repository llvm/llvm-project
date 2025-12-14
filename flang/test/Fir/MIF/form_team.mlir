// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QQmain() attributes {fir.bindc_name = "TEST_FORM_TEAM"} {
  %0 = fir.dummy_scope : !fir.dscope
  %c10 = arith.constant 10 : index
  %1 = fir.alloca !fir.char<1,10> {bindc_name = "err", uniq_name = "_QFEerr"}
  %2:2 = hlfir.declare %1 typeparams %c10 {uniq_name = "_QFEerr"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
  %3 = fir.alloca i32 {bindc_name = "stat", uniq_name = "_QFEstat"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFEstat"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}> {bindc_name = "team", uniq_name = "_QFEteam"}
  %6:2 = hlfir.declare %5 {uniq_name = "_QFEteam"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
  %7 = fir.address_of(@_QQ_QM__fortran_builtinsT__builtin_team_type.DerivedInit) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
  fir.copy %7 to %6#0 no_overlap : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
  %8 = fir.alloca i32 {bindc_name = "team_index", uniq_name = "_QFEteam_index"}
  %9:2 = hlfir.declare %8 {uniq_name = "_QFEteam_index"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %10 = fir.alloca i32 {bindc_name = "team_number", uniq_name = "_QFEteam_number"}
  %11:2 = hlfir.declare %10 {uniq_name = "_QFEteam_number"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %12 = fir.load %11#0 : !fir.ref<i32>
  %13 = fir.embox %6#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
  mif.form_team team_number %12 team_var %13 : (i32, !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> ()
  %14 = fir.load %9#0 : !fir.ref<i32>
  %15 = fir.load %11#0 : !fir.ref<i32>
  %16 = fir.embox %6#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
  mif.form_team team_number %15 team_var %16 new_index %14 : (i32, !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, i32) -> ()
  %17 = fir.load %11#0 : !fir.ref<i32>
  %18 = fir.embox %6#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
  mif.form_team team_number %17 team_var %18 stat %4#0 : (i32, !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<i32>) -> ()
  %19 = fir.embox %2#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
  %20 = fir.load %11#0 : !fir.ref<i32>
  %21 = fir.embox %6#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
  mif.form_team team_number %20 team_var %21 errmsg %19 : (i32, !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.box<!fir.char<1,10>>) -> ()
  return
}
// CHECK: %[[VAL_1:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_2:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_4:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_form_team(%[[TEAM_NUMBER:.*]], %[[VAL_4]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_3]]) : (!fir.ref<i64>, !fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> () 

// CHECK: %[[VAL_5:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_6:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_7:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_form_team(%[[TEAM_NUMBER:.*]], %[[VAL_7]], %[[NEW_INDEX:.*]], %[[VAL_5]], %[[VAL_6]], %[[VAL_6]]) : (!fir.ref<i64>, !fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_8:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_9:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_10:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_form_team(%[[TEAM_NUMBER:.*]], %[[VAL_10]], %[[VAL_8]], %[[START:.*]]#0, %[[VAL_9]], %[[VAL_9]]) : (!fir.ref<i64>, !fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_11:.*]] = fir.embox %[[ERRMSG:.*]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
// CHECK: %[[VAL_12:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_13:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_14:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_15:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: %[[VAL_16:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_form_team(%[[TEAM_NUMBER:.*]], %[[VAL_15]], %[[VAL_12]], %[[VAL_13]], %[[VAL_16]], %[[VAL_14]]) : (!fir.ref<i64>, !fir.box<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
