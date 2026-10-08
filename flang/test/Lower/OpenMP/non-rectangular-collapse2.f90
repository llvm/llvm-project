! Test that non-rectangular collapsed loop nests produce a clear TODO error
! rather than generating incorrect code that crashes at runtime.

! RUN: not %flang_fc1 -emit-fir -fopenmp -fopenmp-version=50 %s -o - 2>&1 | FileCheck %s

! CHECK: not yet implemented: Non-rectangular loop nests with COLLAPSE are not yet supported

! Non-rectangular: lower bound of inner loop depends on outer IV
subroutine non_rect_lb(N)
  implicit none
  integer, intent(in) :: N
  integer :: arr(N,N)
  integer :: i, j

  !$omp parallel do collapse(2)
  do i = 1, N
    do j = i, N
      arr(j,i) = 1
    end do
  end do
end subroutine

