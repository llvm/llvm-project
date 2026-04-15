! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! Verify that collapse(force:N) on non-tightly nested loops (with code
! between loops) does not crash the compiler.

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

! CHECK: func.func @_QPcollapse_force_non_tight
! CHECK: acc.parallel
! CHECK: acc.loop
! CHECK: collapse = [3]
