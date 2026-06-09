! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

subroutine target_update_derived_member_iterator()
  type :: s
    integer :: a(10)
  end type
  type(s) :: x
  integer :: i

  !CHECK: not yet implemented: iterator modifier with derived type member map
  !$omp target update to(iterator(i = 1:10): x%a(i))
end subroutine