! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: iterator modifier with locator outside
! CHECK-SAME: declare mapper variable

module declare_mapper_external_array
  integer :: tbl(100)
  type :: t
    real :: a(100)
  end type

  !$omp declare mapper(mm: t :: v) &
  !$omp& map(iterator(i = 1:100): v%a(i), tbl(i))
end module
