! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! Verify that a compiler directive (e.g. !DIR$ IVDEP) appearing between the
! levels of a collapsed loop nest does not get mistaken for the next nested
! DO CONSTRUCT. The directive is an extra sibling evaluation between the
! NonLabelDoStmt and the inner DoConstruct; the collapse descent must skip
! over it rather than absorbing it as the loop body.

subroutine collapse2_directive_between_loops(n, a)
  integer, intent(in) :: n
  integer :: a(n,n)
  integer :: i, j

  !$acc parallel loop collapse(2) copy(a)
  do i = 1, n
!DIR$ IVDEP
    do j = 1, n
      a(j,i) = 1
    end do
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPcollapse2_directive_between_loops(
! CHECK: acc.parallel
! CHECK: acc.loop combined(parallel)
! CHECK: hlfir.designate
! CHECK: hlfir.assign
! CHECK: acc.yield
! CHECK: collapse([2])

subroutine collapse3_directive_between_loops(n, a)
  integer, intent(in) :: n
  integer :: a(n,n,n)
  integer :: i, j, k

  !$acc parallel loop collapse(3) copy(a)
  do i = 1, n
!DIR$ NOVECTOR
    do j = 1, n
      do k = 1, n
        a(k,j,i) = 1
      end do
    end do
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPcollapse3_directive_between_loops(
! CHECK: acc.parallel
! CHECK: acc.loop combined(parallel)
! CHECK: hlfir.designate
! CHECK: hlfir.assign
! CHECK: acc.yield
! CHECK: collapse([3])
