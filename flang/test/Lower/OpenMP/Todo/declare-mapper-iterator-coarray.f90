! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: iterator modifier with locator outside
! CHECK-SAME: declare mapper variable

module declare_mapper_iterator_coarray
  integer, save :: a(10)[*]
  type :: t
    integer :: x
  end type

  !$omp declare mapper(mm: t :: v) map(iterator(i = 1:10): a(i)[1])
end module
