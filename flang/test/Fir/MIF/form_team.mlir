// RUN: fir-opt --mif-convert %s | FileCheck %s

  func.func @_QQmain() attributes {fir.bindc_name = "TEST_FORM_TEAM"} {
   %0 = fir.dummy_scope : !fir.dscope
    %c10 = arith.constant 10 : index
    %1 = fir.alloca !fir.char<1,10> {bindc_name = "err", uniq_name = "_QFEerr"}
    %2:2 = hlfir.declare %1 typeparams %c10 {uniq_name = "_QFEerr"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
    %3 = fir.alloca i32 {bindc_name = "stat", uniq_name = "_QFEstat"}
    %4:2 = hlfir.declare %3 {uniq_name = "_QFEstat"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5 = fir.address_of(@_QFEteam) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>
    %6:2 = hlfir.declare %5 {uniq_name = "_QFEteam"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>)
    %7 = fir.alloca i32 {bindc_name = "team_index", uniq_name = "_QFEteam_index"}
    %8:2 = hlfir.declare %7 {uniq_name = "_QFEteam_index"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %9 = fir.alloca i32 {bindc_name = "team_number", uniq_name = "_QFEteam_number"}
    %10:2 = hlfir.declare %9 {uniq_name = "_QFEteam_number"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %11 = fir.load %10#0 : !fir.ref<i32>
    mif.form_team team_number %11 team_var %6#0 : (i32, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>) -> ()
    %12 = fir.load %8#0 : !fir.ref<i32>
    %13 = fir.load %10#0 : !fir.ref<i32>
    mif.form_team team_number %13 team_var %6#0 new_index %12 : (i32, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>, i32) -> ()
    %14 = fir.load %10#0 : !fir.ref<i32>
    mif.form_team team_number %14 team_var %6#0 stat %4#0 : (i32, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>, !fir.ref<i32>) -> ()
    %15 = fir.embox %2#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
    %16 = fir.load %10#0 : !fir.ref<i32>
    mif.form_team team_number %16 team_var %6#0 errmsg %15 : (i32, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.info:!fir.box<!fir.ptr<!fir.type<_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type{_QM__fortran_builtinsT__builtin_dummy_team_descriptor_type.__placeholder:i64}>>>}>>, !fir.box<!fir.char<1,10>>) -> ()
    return
  }

// CHECK: %[[VAL_1:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_2:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_4:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.ref<none>
// CHECK: fir.call @_QMprifPprif_form_team(%[[TEAM_NUMBER:.*]], %[[VAL_4]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_3]]) : (!fir.ref<i64>, !fir.ref<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> () 

// CHECK: %[[VAL_5:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_6:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_7:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.ref<none>
// CHECK: fir.call @_QMprifPprif_form_team(%[[TEAM_NUMBER:.*]], %[[VAL_7]], %[[NEW_INDEX:.*]], %[[VAL_5]], %[[VAL_6]], %[[VAL_6]]) : (!fir.ref<i64>, !fir.ref<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_8:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_9:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_10:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.ref<none>
// CHECK: fir.call @_QMprifPprif_form_team(%[[TEAM_NUMBER:.*]], %[[VAL_10]], %[[VAL_8]], %[[START:.*]]#0, %[[VAL_9]], %[[VAL_9]]) : (!fir.ref<i64>, !fir.ref<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_11:.*]] = fir.embox %[[ERRMSG:.*]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
// CHECK: %[[VAL_12:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_13:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_14:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_15:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.ref<none>
// CHECK: %[[VAL_16:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_form_team(%[[TEAM_NUMBER:.*]], %[[VAL_15]], %[[VAL_12]], %[[VAL_13]], %[[VAL_16]], %[[VAL_14]]) : (!fir.ref<i64>, !fir.ref<none>, !fir.ref<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
