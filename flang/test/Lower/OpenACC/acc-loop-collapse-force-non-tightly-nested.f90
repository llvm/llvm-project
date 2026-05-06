! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! Test 1: collapse(force:3) on non-tightly nested loops.
! The inner do k loop should be absorbed by the collapse.
! The prologue (tmp = ...) should be inside the collapsed acc.loop body.

subroutine collapse_force_non_tight(a, n)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: a(n, n, n)
  integer :: i, j, k
  real :: tmp

  !$acc parallel loop collapse(force:3) copy(a)
  do i = 1, n
    do j = 1, n
      tmp = real(i + j)
      do k = 1, n
        a(i, j, k) = tmp + real(k)
      end do
    end do
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPcollapse_force_non_tight
! CHECK: acc.parallel
! Only one acc.loop with 3 IVs
! CHECK: acc.loop combined(parallel)
! Prologue (tmp = i + j) inside the loop body
! CHECK: arith.addi
! Body assignment inside the loop body
! CHECK: hlfir.designate
! CHECK: hlfir.assign
! CHECK: collapse = [3]

! -----

! Test 2: collapse(force:2) with a non-collapsed inner !$acc loop.
! The do k loop has its own !$acc loop directive and should NOT be
! absorbed by the collapse — it must remain as a separate acc.loop.

subroutine collapse_force_with_inner_acc_loop(a, n)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: a(n, n, n)
  integer :: i, j, k

  !$acc parallel loop collapse(force:2) copy(a)
  do i = 1, n
    do j = 1, n
      !$acc loop
      do k = 1, n
        a(i, j, k) = real(i + j + k)
      end do
    end do
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPcollapse_force_with_inner_acc_loop
! CHECK: acc.parallel
! Outer collapsed acc.loop
! CHECK: acc.loop combined(parallel)
! Inner acc.loop for k should remain (not absorbed by collapse)
! CHECK: acc.loop
! CHECK: acc.yield
! Outer has collapse = [2]
! CHECK: collapse = [2]

! -----

! Test 3: collapse(force:3) with statements before an inner !$acc loop.
! The do k loop is absorbed by the collapse (3 IVs), but do l with
! its own !$acc loop directive should remain as a separate acc.loop.

subroutine collapse_force3_with_inner_acc_loop(a, n)
  implicit none
  integer, intent(in) :: n
  real, intent(inout) :: a(n, n, n, n)
  integer :: i, j, k, l
  real :: tmp

  !$acc parallel loop collapse(force:3) copy(a)
  do i = 1, n
    do j = 1, n
      tmp = real(i + j)
      do k = 1, n
        a(i, j, k, 1) = tmp
        !$acc loop
        do l = 1, n
          a(i, j, k, l) = a(i, j, k, l) + real(l)
        end do
      end do
    end do
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPcollapse_force3_with_inner_acc_loop
! CHECK: acc.parallel
! Outer collapsed acc.loop with 3 IVs
! CHECK: acc.loop combined(parallel)
! Prologue (tmp = i + j) inside the collapsed body
! CHECK: arith.addi
! Statement before inner acc loop (a(i,j,k,1) = tmp)
! CHECK: hlfir.assign
! Inner acc.loop for l should remain
! CHECK: acc.loop
! CHECK: acc.yield
! Outer has collapse = [3]
! CHECK: collapse = [3]
