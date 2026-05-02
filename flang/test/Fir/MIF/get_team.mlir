// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (git@github.com:SiPearl/llvm-project.git 666e4313ebc03587f27774139ad8f780bac15c3e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() attributes {fir.bindc_name = "TEST_FORM_TEAM"} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@_QMiso_fortran_envECcurrent_team) : !fir.ref<i32>
    %2:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECcurrent_team"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.address_of(@_QMiso_fortran_envECinitial_team) : !fir.ref<i32>
    %4:2 = hlfir.declare %3 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECinitial_team"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFEn"}
    %6:2 = hlfir.declare %5 {uniq_name = "_QFEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %7 = fir.address_of(@_QMiso_fortran_envECparent_team) : !fir.ref<i32>
    %8:2 = hlfir.declare %7 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMiso_fortran_envECparent_team"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %9 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}> {bindc_name = "result_team", uniq_name = "_QFEresult_team"}
    %10:2 = hlfir.declare %9 {uniq_name = "_QFEresult_team"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
    %11 = fir.address_of(@_QQ_QM__fortran_builtinsT__builtin_team_type.DerivedInit) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    fir.copy %11 to %10#0 no_overlap : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %12 = mif.get_team : () -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %13:2 = hlfir.declare %12 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
    %false = arith.constant false
    %14 = hlfir.as_expr %13#0 move %false : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, i1) -> !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    hlfir.assign %14 to %10#0 : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    hlfir.destroy %14 : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %c-2_i32 = arith.constant -2 : i32
    %15 = mif.get_team level %c-2_i32 : (i32) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %16:2 = hlfir.declare %15 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
    %false_0 = arith.constant false
    %17 = hlfir.as_expr %16#0 move %false_0 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, i1) -> !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    hlfir.assign %17 to %10#0 : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    hlfir.destroy %17 : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %c-1_i32 = arith.constant -1 : i32
    %18 = mif.get_team level %c-1_i32 : (i32) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %19:2 = hlfir.declare %18 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
    %false_1 = arith.constant false
    %20 = hlfir.as_expr %19#0 move %false_1 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, i1) -> !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    hlfir.assign %20 to %10#0 : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    hlfir.destroy %20 : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %c-3_i32 = arith.constant -3 : i32
    %21 = mif.get_team level %c-3_i32 : (i32) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %22:2 = hlfir.declare %21 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
    %false_2 = arith.constant false
    %23 = hlfir.as_expr %22#0 move %false_2 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, i1) -> !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    hlfir.assign %23 to %10#0 : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    hlfir.destroy %23 : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %24 = fir.load %6#0 : !fir.ref<i32>
    %25 = mif.get_team level %24 : (i32) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %26:2 = hlfir.declare %25 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
    %false_3 = arith.constant false
    %27 = hlfir.as_expr %26#0 move %false_3 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, i1) -> !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    hlfir.assign %27 to %10#0 : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    hlfir.destroy %27 : !hlfir.expr<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    return
  }
}

// CHECK: %[[VAL_1:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[RESULT:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_get_team(%[[VAL_1]], %[[RESULT]]) : (!fir.ref<i32>, !fir.box<none>) -> ()

// CHECK: %[[RESULT:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_get_team(%[[INIT:.*]], %[[RESULT]]) : (!fir.ref<i32>, !fir.box<none>) -> ()

// CHECK: %[[RESULT:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_get_team(%[[CURRENT:.*]], %[[RESULT]]) : (!fir.ref<i32>, !fir.box<none>) -> ()

// CHECK: %[[RESULT:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_get_team(%[[PARENT:.*]], %[[RESULT]]) : (!fir.ref<i32>, !fir.box<none>) -> ()

// CHECK: %[[RESULT:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_get_team(%[[VAL_N:.*]], %[[RESULT]]) : (!fir.ref<i32>, !fir.box<none>) -> ()
