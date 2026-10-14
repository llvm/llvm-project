! COLLAPSE over a nested loop-transforming construct (e.g. TILE) is accepted by
! semantics but not yet supported in lowering. Check for a "not yet
! implemented" message instead of a compiler crash.

! RUN: not %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -o - %s 2>&1 | FileCheck %s

subroutine collapse_tile(n, x)
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, j

  !$omp do collapse(4)
  !$omp tile sizes(2, 2)
  do i = 1, n
    do j = 1, n
      x = x + i + j
    end do
  end do
end subroutine

! CHECK: not yet implemented: Collapsing a loop nest that contains a loop-transforming construct
