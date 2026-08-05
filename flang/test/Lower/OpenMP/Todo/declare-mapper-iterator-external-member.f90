! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: iterator modifier with derived type member map

module declare_mapper_external_member
  type :: t
    integer :: a(10)
  end type
  type(t) :: w

  !$omp declare mapper(m: t :: v) map(iterator(i = 1:10): w%a(i))
end module
