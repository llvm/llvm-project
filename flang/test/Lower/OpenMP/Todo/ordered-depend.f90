!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! Issue #198972: a standalone ordered construct using the (pre-5.2)
! depend(source) / depend(sink:) spelling must reach the "not yet
! implemented" path instead of crashing during construct decomposition. The
! depend(source|sink:) spelling is represented internally as a doacross
! clause, which decomposition only accepts from OpenMP 5.2, while the construct
! itself is valid (using this spelling) since OpenMP 4.5.

!CHECK: not yet implemented: OMPD_ordered
subroutine f00
  integer :: i
  !$omp do ordered(1)
  do i = 1, 10
    !$omp ordered depend(source)
    !$omp ordered depend(sink: i - 1)
  end do
  !$omp end do
end

