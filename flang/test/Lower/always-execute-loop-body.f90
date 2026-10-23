! RUN: %flang_fc1 -mmlir --always-execute-loop-body -emit-hlfir %s -o - | FileCheck %s

! Given the flag `--always-execute-loop-body` the compiler emits an extra
! code to change the trip count, test tries to verify the extra emitted HLFIR.
!
! The trip count only exists on the unstructured lowering path, so the loop has
! to stay unstructured for the flag to have anything to act on: the GOTO leaves
! the loop, which keeps it out of the structured form.

! CHECK-LABEL: func.func @_QPsome
subroutine some()
  integer :: i

  ! CHECK: %[[TRIP:.*]] = fir.alloca i32
  ! CHECK: %[[I_RAW:.*]] = fir.alloca i32 <{bindc_name = "i"
  ! CHECK: %[[I:.*]]:2 = hlfir.declare %[[I_RAW]]
  ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[TRIP_COUNT:.*]] = arith.select {{.*}}, %c1{{.*}}, {{.*}} : i32
  ! CHECK: fir.store %[[TRIP_COUNT]] to %[[TRIP]] : !fir.ref<i32>
  ! CHECK: %[[LOADED_TRIP:.*]] = fir.load %[[TRIP]] : !fir.ref<i32>
  ! CHECK: %[[CMP:.*]] = arith.cmpi sgt, %[[LOADED_TRIP]], %c0{{.*}} : i32
  ! CHECK: cf.cond_br %[[CMP]]
  do i=4,1,1
    goto 9
  end do
9 return
end
