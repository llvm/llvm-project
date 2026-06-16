! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! Verify collapse(force:2) sinks prologue/epilogue and eliminates
! the inner loop (absorbed by the outer collapse).

subroutine collapse_force_sink(n, m)
  integer, intent(in) :: n, m
  real, dimension(n,m) :: a
  real, dimension(n) :: bb, cc
  integer :: i, j

  !$acc parallel loop collapse(force:2)
  do i = 1, n
    bb(i) = 4.2     ! prologue (between loops)
    do j = 1, m
      a(i,j) = a(i,j) + 2.0
    end do
    cc(i) = 7.3     ! epilogue (after inner loop)
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPcollapse_force_sink(
! CHECK: acc.parallel
! Only one acc.loop with collapse = [2], inner loop absorbed
! CHECK: acc.loop combined(parallel)
! Prologue (4.2), body (2.0 add), epilogue (7.3) all inside
! CHECK: arith.constant 4.200000e+00
! CHECK: hlfir.assign
! CHECK: arith.constant 2.000000e+00
! CHECK: hlfir.assign
! CHECK: arith.constant 7.300000e+00
! CHECK: hlfir.assign
! CHECK: collapse = [2]
