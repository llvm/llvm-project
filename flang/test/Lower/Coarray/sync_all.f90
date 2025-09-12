! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

program test_sync_all
  implicit none
  ! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.
 
  ! COARRAY: %[[ERRMSG:.*]]:2 = hlfir.declare %[[VAL_1:.*]] typeparams %[[C_128:.*]] {uniq_name = "_QFEerror_message"} : (!fir.ref<!fir.char<1,128>>, index) -> (!fir.ref<!fir.char<1,128>>, !fir.ref<!fir.char<1,128>>)
  ! COARRAY: %[[STAT:.*]]:2 = hlfir.declare %[[VAL_2:.*]] {uniq_name = "_QFEsync_status"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer sync_status
  character(len=128) :: error_message

  ! COARRAY: %[[VAL_3:.*]] = fir.absent !fir.ref<i32>
  ! COARRAY: %[[VAL_4:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: %[[VAL_5:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: fir.call @_QMprifPprif_sync_all(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]]) fastmath<contract> : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  sync all

  ! COARRAY: %[[VAL_6:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: %[[VAL_7:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: fir.call @_QMprifPprif_sync_all(%[[STAT]]#0, %[[VAL_6]], %[[VAL_7]]) fastmath<contract> : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  sync all(stat=sync_status)
  
  ! COARRAY: %[[VAL_8:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
  ! COARRAY: %[[VAL_9:.*]] = fir.absent !fir.ref<i32>
  ! COARRAY: %[[VAL_10:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: %[[VAL_11:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
  ! COARRAY: fir.call @_QMprifPprif_sync_all(%[[VAL_9]], %[[VAL_11]], %[[VAL_10]]) fastmath<contract> : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  sync all(                  errmsg=error_message)
  
  ! COARRAY: %[[VAL_12:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
  ! COARRAY: %[[VAL_13:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
  ! COARRAY: fir.call @_QMprifPprif_sync_all(%[[STAT]]#0, %[[VAL_14]], %[[VAL_13]]) fastmath<contract> : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  sync all(stat=sync_status, errmsg=error_message)

end program test_sync_all
