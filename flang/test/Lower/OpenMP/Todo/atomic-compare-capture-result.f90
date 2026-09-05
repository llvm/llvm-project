! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! The comparison-result form of atomic compare captures the boolean outcome of
! the comparison into `r`:
!   r = x == e
!   if (r) x = d
! This is accepted by semantics, but the atomic analysis does not record the
! assignment to `r`, so lowering would silently drop it and never write `r`.
! Ensure it is diagnosed as "not yet implemented" instead of producing code that
! leaves `r` unwritten.

! CHECK: not yet implemented: atomic compare capturing the comparison result
subroutine f(x, e, d, r)
  integer :: x, e, d
  logical :: r
  !$omp atomic compare
  r = x == e
  if (r) x = d
  !$omp end atomic
end subroutine
