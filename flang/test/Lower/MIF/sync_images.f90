! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

program test_sync_images
  implicit none
  ! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.
 
  ! COARRAY: %[[ERRMSG:.*]]:2 = hlfir.declare %[[VAL_1:.*]] typeparams %[[C_128:.*]] {uniq_name = "_QFEerror_message"} : (!fir.ref<!fir.char<1,128>>, index) -> (!fir.ref<!fir.char<1,128>>, !fir.ref<!fir.char<1,128>>)
  ! COARRAY: %[[ME:.*]]:2 = hlfir.declare %[[VAL_3:.*]] {uniq_name = "_QFEme"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! COARRAY: %[[STAT:.*]]:2 = hlfir.declare %[[VAL_2:.*]] {uniq_name = "_QFEsync_status"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  integer sync_status, me
  character(len=128) :: error_message

  ! COARRAY: %[[VAL_1:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
  ! COARRAY: mif.sync_images stat %[[STAT]]#0 errmsg %[[VAL_1]] : (!fir.ref<i32>, !fir.box<!fir.char<1,128>>)
  sync images(*, stat=sync_status, errmsg=error_message)

  ! COARRAY: %[[VAL_2:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
  ! COARRAY: %[[VAL_3:.*]] = fir.embox %[[ME]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! COARRAY: mif.sync_images image_set %[[VAL_3]] stat %[[STAT]]#0 errmsg %[[VAL_2]] : (!fir.box<i32>, !fir.ref<i32>, !fir.box<!fir.char<1,128>>)
  sync images(me,   stat=sync_status, errmsg=error_message)

  ! COARRAY: %[[VAL_4:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
  ! COARRAY: %[[VAL_5:.*]] = fir.embox %[[IMG_SET:.*]]#0(%[[SHAPE_1:.*]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
  ! COARRAY: mif.sync_images image_set %[[VAL_5]] stat %[[STAT]]#0 errmsg %[[VAL_4]] : (!fir.box<!fir.array<1xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,128>>)
  sync images([1],  stat=sync_status, errmsg=error_message)
  
  ! COARRAY: mif.sync_images : ()
  sync images(*)
  
  ! COARRAY: %[[VAL_6:.*]] = fir.embox %[[ME]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! COARRAY: mif.sync_images image_set %[[VAL_6]] : (!fir.box<i32>)
  sync images(me)
  
  ! COARRAY: %[[VAL_7:.*]] = fir.embox %[[IMG_SET:.*]]#0(%[[SHAPE_3:.*]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
  ! COARRAY: mif.sync_images image_set %[[VAL_7]] : (!fir.box<!fir.array<1xi32>>)
  sync images([1])

end program test_sync_images
