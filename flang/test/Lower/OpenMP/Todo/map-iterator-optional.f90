! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

subroutine target_enter_data_optional_iterator(a, n)
  integer, optional, intent(inout) :: a(:)
  integer, intent(in) :: n
  integer :: i

  ! CHECK: not yet implemented: iterator modifier with optional locator
  !$omp target enter data map(iterator(i = 1:n), to: a(i))
end subroutine
