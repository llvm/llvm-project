// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (git@github.com:SiPearl/llvm-project.git 666e4313ebc03587f27774139ad8f780bac15c3e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
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
  }

// CHECK: %[[VAL_1:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_team_number(%[[VAL_1]], %[[RESULT:.*]]) : (!fir.box<none>, !fir.ref<i64>) -> ()
// CHECK: %[[VAL_2:.*]] = fir.load %[[RESULT]] : !fir.ref<i64>

// CHECK: %[[VAL_3:.*]] = fir.absent !fir.box<none>
// CHECK: fir.call @_QMprifPprif_team_number(%[[VAL_3]], %[[RESULT:.*]]) : (!fir.box<none>, !fir.ref<i64>) -> ()
// CHECK: %[[VAL_4:.*]] = fir.load %[[RESULT]] : !fir.ref<i64>
