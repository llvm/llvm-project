! RUN: bbc --always-execute-loop-body --emit-fir %s -o - | FileCheck %s

! Given the flag `--always-execute-loop-body` the compiler emits an extra
! code to change to tripcount, test tries to verify the extra emitted FIR.

! CHECK-LABEL: func @_QPsome
subroutine some()
  integer :: i

  ! CHECK: [[tripcount:%[0-9]+]] = divi_signed
  ! CHECK: [[zero:%c0_i32]] = constant 0 : i32
  ! CHECK: [[cmp:%5]] = cmpi "sle", [[tripcount]], [[zero]] : i32
  ! CHECK: [[one:%c1_i32_1]] = constant 1 : i32
  ! CHECK: [[newtripcount:%6]] = select [[cmp]], [[one]], [[tripcount]] : i32
  ! CHECK: fir.store [[newtripcount]] to %0 : !fir.ref<i32>
  do i=4,1,1
    stop 2
  end do
  return
end

