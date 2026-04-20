! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.


program event_test
  use iso_fortran_env, only: event_type
  implicit none
  
  type(event_type)  :: data_ready[*]
  integer :: me

  me = this_image()

  if (me == 2) then
     event post(data_ready[1])
  else if (me == 1) then
     event wait(data_ready, UNTIL_COUNT=1)
  end if
end program

! COARRAY: %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
! COARRAY: %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
! COARRAY: %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY: %[[VAL_3:.*]] = fir.address_of(@_QFEdata_ready) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY: %c1_i64 = arith.constant 1 : i64
! COARRAY: %c0 = arith.constant 0 : index
! COARRAY: %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_1]], %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! COARRAY: fir.store %c1_i64 to %[[VAL_4]] : !fir.ref<i64>
! COARRAY: %[[VAL_5:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! COARRAY: %[[VAL_6:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! COARRAY: mif.alloc_coarray %[[VAL_3]] lcobounds %[[VAL_5]] ucobounds %[[VAL_6]] {uniq_name = "_QFEdata_ready"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
! COARRAY: %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFEdata_ready"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>)
! COARRAY: %[[VAL_8:.*]] = fir.alloca i32 {bindc_name = "me", uniq_name = "_QFEme"}
! COARRAY: %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFEme"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! COARRAY: %[[VAL_10:.*]] = mif.this_image : () -> i32
! COARRAY: hlfir.assign %[[VAL_10]] to %[[VAL_9]]#0 : i32, !fir.ref<i32>
! COARRAY: %[[VAL_11:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
! COARRAY: %c2_i32 = arith.constant 2 : i32
! COARRAY: %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_11]], %c2_i32 : i32
! COARRAY: fir.if %[[VAL_12]] {
! COARRAY:   %[[VAL_13:.*]] = hlfir.designate %[[VAL_7]]#0   : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
! COARRAY:   %[[VAL_14:.*]] = fir.embox %[[VAL_13]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> !fir.box<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>, corank:1>
! COARRAY:   %c1_i64_0 = arith.constant 1 : i64
! COARRAY:   mif.event_post %[[VAL_14]][%c1_i64_0] : (!fir.box<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>, corank:1>, i64) -> ()
! COARRAY: } else {
! COARRAY:   %[[VAL_15:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
! COARRAY:   %c1_i32 = arith.constant 1 : i32
! COARRAY:   %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_15]], %c1_i32 : i32
! COARRAY:   fir.if %[[VAL_16]] {
! COARRAY:     %c1_i32_0 = arith.constant 1 : i32
! COARRAY:     mif.event_wait %[[VAL_7]]#0 until_count %c1_i32_0 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, i32) -> ()
! COARRAY:   }
! COARRAY: }
! COARRAY: return
