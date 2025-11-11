! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! Verify collapse(force:2) sinks prologue (between loops) and epilogue (after inner loop)
! into the acc.loop region body.

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

! CHECK: func.func @_QPcollapse_force_sink(
! CHECK: acc.parallel
! Ensure outer acc.loop is combined(parallel)
! CHECK: acc.loop combined(parallel)
! Prologue: constant 4.2 and an assign before inner loop
! CHECK: arith.constant 4.200000e+00
! CHECK: hlfir.assign
! Inner loop and its body include 2.0 add and an assign
! CHECK: acc.loop
! CHECK: arith.constant 2.000000e+00
! CHECK: arith.addf
! CHECK: hlfir.assign
! Epilogue: constant 7.3 and an assign after inner loop
! CHECK: arith.constant 7.300000e+00
! CHECK: hlfir.assign
! And the outer acc.loop has collapse = [2]
! CHECK: } attributes {collapse = [2]


