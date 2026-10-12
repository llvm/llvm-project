! Test that disabling the kernels-loop DO CONCURRENT independence extension
! lowers the loop as auto.
!
! RUN: %flang_fc1 -fopenacc \
! RUN:   -fno-openacc-acc-kernels-do-concurrent-independent \
! RUN:   -emit-hlfir %s -o - | FileCheck %s

subroutine kernels_loop_do_concurrent(n, x)
  integer :: n, i
  real :: x(n)
  !$acc kernels loop
  do concurrent (i = 1:n)
    x(i) = real(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPkernels_loop_do_concurrent
! CHECK: acc.kernels combined(loop)
! CHECK: acc.loop combined(kernels)
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
