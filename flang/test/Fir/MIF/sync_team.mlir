// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (git@github.com:SiPearl/llvm-project.git 6c5d1dd436b2ce3e45d7b19a64028198d4fe9b35)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() attributes {fir.bindc_name = "TEST_SYNC_TEAM"} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@_QFEerror_message) : !fir.ref<!fir.char<1,128>>
    %c128 = arith.constant 128 : index
    %2:2 = hlfir.declare %1 typeparams %c128 {uniq_name = "_QFEerror_message"} : (!fir.ref<!fir.char<1,128>>, index) -> (!fir.ref<!fir.char<1,128>>, !fir.ref<!fir.char<1,128>>)
    %3 = fir.alloca i32 {bindc_name = "sync_status", uniq_name = "_QFEsync_status"}
    %4:2 = hlfir.declare %3 {uniq_name = "_QFEsync_status"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5 = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}> {bindc_name = "team", uniq_name = "_QFEteam"}
    %6:2 = hlfir.declare %5 {uniq_name = "_QFEteam"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>)
    %7 = fir.address_of(@_QQ_QM__fortran_builtinsT__builtin_team_type.DerivedInit) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    fir.copy %7 to %6#0 no_overlap : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %8 = fir.embox %6#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    mif.sync_team %8 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> ()
    %9 = fir.embox %6#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    mif.sync_team %9 stat %4#0 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<i32>) -> ()
    %10 = fir.embox %6#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %11 = fir.embox %2#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
    mif.sync_team %10 errmsg %11 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.box<!fir.char<1,128>>) -> ()
    %12 = fir.embox %6#0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>
    %13 = fir.embox %2#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
    mif.sync_team %12 stat %4#0 errmsg %13 : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_team_type{_QM__fortran_builtinsT__builtin_team_type.__id:i64}>>, !fir.ref<i32>, !fir.box<!fir.char<1,128>>) -> ()
    return
  }
  fir.global internal @_QFEerror_message : !fir.char<1,128> {
    %0 = fir.zero_bits !fir.char<1,128>
    fir.has_value %0 : !fir.char<1,128>
  }
}

// CHECK: %[[ERRMSG:.*]]:2 = hlfir.declare %[[E:.*]] typeparams %[[C_128:.*]] {uniq_name = "_QFEerror_message"} : (!fir.ref<!fir.char<1,128>>, index) -> (!fir.ref<!fir.char<1,128>>, !fir.ref<!fir.char<1,128>>)
// CHECK: %[[STAT:.*]]:2 = hlfir.declare %[[S:.*]] {uniq_name = "_QFEsync_status"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

// CHECK: %[[VAL_1:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_2:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[TEAM_2:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_sync_team(%[[TEAM_2]], %[[VAL_2]], %[[VAL_1]], %[[VAL_1]]) : (!fir.box<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[TEAM_2:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: fir.call @_QMprifPprif_sync_team(%[[TEAM_2]], %[[STAT]]#0, %[[VAL_3]], %[[VAL_3]]) : (!fir.box<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_4:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
// CHECK: %[[VAL_5:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_6:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[TEAM_2:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_sync_team(%[[TEAM_2]], %[[VAL_6]], %[[VAL_7]], %[[VAL_5]]) : (!fir.box<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_8:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
// CHECK: %[[VAL_9:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[TEAM_2:.*]] = fir.convert %[[TEAM:.*]] : ({{.*}}) -> !fir.box<none>
// CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_sync_team(%[[TEAM_2]], %[[STAT]]#0, %[[VAL_10]], %[[VAL_9]]) : (!fir.box<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
