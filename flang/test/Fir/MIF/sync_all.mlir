// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (git@github.com:SiPearl/llvm-project.git 666e4313ebc03587f27774139ad8f780bac15c3e)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() attributes {fir.bindc_name = "TEST_SYNC_ALL"} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@_QFEerror_message) : !fir.ref<!fir.char<1,128>>
    %c128 = arith.constant 128 : index
    %2:2 = hlfir.declare %1 typeparams %c128 {uniq_name = "_QFEerror_message"} : (!fir.ref<!fir.char<1,128>>, index) -> (!fir.ref<!fir.char<1,128>>, !fir.ref<!fir.char<1,128>>)
    %3 = fir.alloca i32 {bindc_name = "sync_status", uniq_name = "_QFEsync_status"}
    %4:2 = hlfir.declare %3 {uniq_name = "_QFEsync_status"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    mif.sync_all : () -> ()
    mif.sync_all stat %4#0 : (!fir.ref<i32>) -> ()
    %5 = fir.embox %2#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
    mif.sync_all errmsg %5 : (!fir.box<!fir.char<1,128>>) -> ()
    %6 = fir.embox %2#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
    mif.sync_all stat %4#0 errmsg %6 : (!fir.ref<i32>, !fir.box<!fir.char<1,128>>) -> ()
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
// CHECK: fir.call @_QMprifPprif_sync_all(%[[VAL_2]], %[[VAL_1]], %[[VAL_1]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_sync_all(%[[STAT]]#0, %[[VAL_3]], %[[VAL_3]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_4:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
// CHECK: %[[VAL_5:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_6:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_sync_all(%[[VAL_6]], %[[VAL_7]], %[[VAL_5]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_8:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
// CHECK: %[[VAL_9:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_sync_all(%[[STAT]]#0, %[[VAL_10]], %[[VAL_9]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
