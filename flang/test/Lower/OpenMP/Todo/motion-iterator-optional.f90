! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

subroutine target_update_optional_iterator(a, n)
  integer, allocatable, optional, intent(inout) :: a(:)
  integer, intent(in) :: n
  integer :: i

  ! CHECK: not yet implemented: iterator modifier with optional locator
  !$omp target update to(iterator(i = 1:n): a(i))
end subroutine
