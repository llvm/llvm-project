! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

program test_sync_memory
  implicit none
  ! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.
 
  ! COARRAY: %[[ERRMSG:.*]]:2 = hlfir.declare %[[VAL_1:.*]] typeparams %[[C_128:.*]] {uniq_name = "_QFEerror_message"} : (!fir.ref<!fir.char<1,128>>, index) -> (!fir.ref<!fir.char<1,128>>, !fir.ref<!fir.char<1,128>>)
  ! COARRAY: %[[STAT:.*]]:2 = hlfir.declare %[[VAL_2:.*]] {uniq_name = "_QFEsync_status"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer sync_status
  character(len=128) :: error_message

  ! COARRAY: mif.sync_memory : () 
  sync memory

  ! COARRAY: mif.sync_memory stat %[[STAT]]#0 : (!fir.ref<i32>)
  sync memory(stat=sync_status)
  
  ! COARRAY: %[[VAL_1:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
  ! COARRAY: mif.sync_memory errmsg %[[VAL_1]] : (!fir.box<!fir.char<1,128>>)
  sync memory(                  errmsg=error_message)
  
  ! COARRAY: %[[VAL_2:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
  ! COARRAY: mif.sync_memory stat %[[STAT]]#0 errmsg %[[VAL_2]] : (!fir.ref<i32>, !fir.box<!fir.char<1,128>>)
  sync memory(stat=sync_status, errmsg=error_message)

end program test_sync_memory
