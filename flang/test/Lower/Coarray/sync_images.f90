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
  ! COARRAY: %[[VAL_2:.*]] = fir.absent !fir.box<!fir.array<?xi32>> 
  ! COARRAY: %[[VAL_3:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: %[[VAL_4:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
  ! COARRAY: fir.call @_QMprifPprif_sync_images(%[[VAL_2]], %[[STAT]]#0, %[[VAL_4]], %[[VAL_3]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  sync images(*, stat=sync_status, errmsg=error_message)

  ! COARRAY: %[[VAL_5:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
  ! COARRAY: %[[VAL_6:.*]] = fir.embox %[[ME]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! COARRAY: %[[VAL_7:.*]] = fir.rebox %[[VAL_6]](%[[SHAPE:.*]]) : (!fir.box<i32>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
  ! COARRAY: %[[VAL_8:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: %[[VAL_9:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<!fir.array<?xi32>> 
  ! COARRAY: %[[VAL_10:.*]] = fir.convert %[[VAL_5]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
  ! COARRAY: fir.call @_QMprifPprif_sync_images(%[[VAL_9]], %[[STAT]]#0, %[[VAL_10]], %[[VAL_8]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  sync images(me,   stat=sync_status, errmsg=error_message)

  ! COARRAY: %[[VAL_11:.*]] = fir.embox %[[ERRMSG]]#0 : (!fir.ref<!fir.char<1,128>>) -> !fir.box<!fir.char<1,128>>
  ! COARRAY: %[[VAL_12:.*]] = fir.embox %[[IMG_SET:.*]]#0(%[[SHAPE_1:.*]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
  ! COARRAY: %[[VAL_13:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<!fir.array<?xi32>>
  ! COARRAY: %[[VAL_15:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.char<1,128>>) -> !fir.box<!fir.char<1,?>>
  ! COARRAY: fir.call @_QMprifPprif_sync_images(%[[VAL_14]], %[[STAT]]#0, %[[VAL_15]], %[[VAL_13]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  sync images([1],  stat=sync_status, errmsg=error_message)
  
  ! COARRAY: %[[VAL_17:.*]] = fir.absent !fir.ref<i32>
  ! COARRAY: %[[VAL_18:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: %[[VAL_19:.*]] = fir.absent !fir.box<!fir.array<?xi32>> 
  ! COARRAY: %[[VAL_20:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: fir.call @_QMprifPprif_sync_images(%[[VAL_19]], %[[VAL_17]], %[[VAL_18]], %[[VAL_20]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  sync images(*)
  
  ! COARRAY: %[[VAL_23:.*]] = fir.absent !fir.ref<i32>
  ! COARRAY: %[[VAL_24:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: %[[VAL_21:.*]] = fir.embox %[[ME]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
  ! COARRAY: %[[VAL_22:.*]] = fir.rebox %[[VAL_21]](%[[SHAPE_2:.*]]) : (!fir.box<i32>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
  ! COARRAY: %[[VAL_25:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: %[[VAL_26:.*]] = fir.convert %[[VAL_22]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<!fir.array<?xi32>> 
  ! COARRAY: fir.call @_QMprifPprif_sync_images(%[[VAL_26]], %[[VAL_23]], %[[VAL_24]], %[[VAL_25]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  sync images(me)
  
  ! COARRAY: %[[VAL_28:.*]] = fir.absent !fir.ref<i32>
  ! COARRAY: %[[VAL_29:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: %[[VAL_27:.*]] = fir.embox %[[IMG_SET:.*]]#0(%[[SHAPE_3:.*]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
  ! COARRAY: %[[VAL_30:.*]] = fir.absent !fir.box<!fir.char<1,?>>
  ! COARRAY: %[[VAL_31:.*]] = fir.convert %[[VAL_27]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<!fir.array<?xi32>>
  ! COARRAY: fir.call @_QMprifPprif_sync_images(%[[VAL_31]], %[[VAL_28]], %[[VAL_29]], %[[VAL_30]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
  sync images([1])

end program test_sync_images
