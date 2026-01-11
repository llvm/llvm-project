! RUN: bbc --always-execute-loop-body --emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -mmlir --always-execute-loop-body -emit-fir %s -o - | FileCheck %s

! Given the flag `--always-execute-loop-body` the compiler emits an extra
! code to change to tripcount, test tries to verify the extra emitted FIR.

! CHECK-LABEL: func @_QPsome
subroutine some()
  integer :: i

  ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: %[[TRIP:.*]] = fir.alloca i32
  ! CHECK: fir.store %[[C1]] to %[[TRIP]] : !fir.ref<i32>
  ! CHECK: %[[LOADED_TRIP:.*]] = fir.load %[[TRIP]] : !fir.ref<i32>
  ! CHECK: %[[CMP:.*]] = arith.cmpi sgt, %[[LOADED_TRIP]], %c0{{.*}} : i32
  ! CHECK: cf.cond_br %[[CMP]]
  do i=4,1,1
    stop 2
  end do
  return
end
