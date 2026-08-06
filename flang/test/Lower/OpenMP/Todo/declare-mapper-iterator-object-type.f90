! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: object type not supported by iterator modifier

module declare_mapper_iterator_object_type
  type :: t
    complex :: c(10)
  end type

  !$omp declare mapper(mm: t :: v) map(iterator(i = 1:10): v%c(i)%re)
end module
