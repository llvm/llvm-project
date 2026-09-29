// RUN: fir-opt --mif-convert %s | FileCheck %s

  func.func @_QQmain() attributes {fir.bindc_name = "TEST_sync_memory"} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.address_of(@_QFEerror_message) : !fir.ref<!fir.char<1,128>>
    %c128 = arith.constant 128 : index
    %2:2 = hlfir.declare %1 typeparams %c128 {uniq_name = "_QFEerror_message"} : (!fir.ref<!fir.char<1,128>>, index) -> (!fir.ref<!fir.char<1,128>>, !fir.ref<!fir.char<1,128>>)
    %3 = fir.alloca i32 {bindc_name = "sync_status", uniq_name = "_QFEsync_status"}
    %4:2 = hlfir.declare %3 {uniq_name = "_QFEsync_status"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    mif.sync_memory : () -> ()
    mif.sync_memory stat %4#0 : (!fir.ref<i32>) -> ()
    %5 = fir.embox %2#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
    mif.sync_memory errmsg %5 : (!fir.box<!fir.char<1,128>>) -> ()
    %6 = fir.embox %2#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
    mif.sync_memory stat %4#0 errmsg %6 : (!fir.ref<i32>, !fir.box<!fir.char<1,128>>) -> ()
    return
  }
  fir.global internal @_QFEerror_message : !fir.char<1,128> {
    %0 = fir.zero_bits !fir.char<1,128>
    fir.has_value %0 : !fir.char<1,128>
  }


// CHECK: %[[ERRMSG:.*]]:2 = hlfir.declare %[[E:.*]] typeparams %[[C_128:.*]] {uniq_name = "_QFEerror_message"} : (!fir.ref<!fir.char<1,128>>, index) -> (!fir.ref<!fir.char<1,128>>, !fir.ref<!fir.char<1,128>>)
// CHECK: %[[STAT:.*]]:2 = hlfir.declare %[[S:.*]] {uniq_name = "_QFEsync_status"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

// CHECK: %[[VAL_1:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_2:.*]] = fir.absent !fir.ref<i32>
// CHECK: fir.call @_QMprifPprif_sync_memory(%[[VAL_2]], %[[VAL_1]], %[[VAL_1]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_sync_memory(%[[STAT]]#0, %[[VAL_3]], %[[VAL_3]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_4:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
// CHECK: %[[VAL_5:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_6:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_sync_memory(%[[VAL_6]], %[[VAL_7]], %[[VAL_5]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()

// CHECK: %[[VAL_8:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
// CHECK: %[[VAL_9:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_sync_memory(%[[STAT]]#0, %[[VAL_10]], %[[VAL_9]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
