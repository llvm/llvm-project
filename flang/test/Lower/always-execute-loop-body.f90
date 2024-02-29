! RUN: bbc --always-execute-loop-body --emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -mmlir --always-execute-loop-body -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! Given the flag `--always-execute-loop-body` the compiler emits an extra
! code to change to tripcount, test tries to verify the extra emitted FIR.

! CHECK-LABEL: func @_QPsome
subroutine some()
  integer :: i

  ! CHECK: [[tripcount:%[0-9]+]] = arith.divsi
  ! CHECK: [[one:%c1_i32[_0-9]*]] = arith.constant 1 : i32
  ! CHECK: [[cmp:%[0-9]+]] = arith.cmpi slt, [[tripcount]], [[one]] : i32
  ! CHECK: [[newtripcount:%[0-9]+]] = arith.select [[cmp]], [[one]], [[tripcount]] : i32
  ! CHECK: fir.store [[newtripcount]] to %{{[0-9]+}} : !fir.ref<i32>
  do i=4,1,1
    stop 2
  end do
  return
end
