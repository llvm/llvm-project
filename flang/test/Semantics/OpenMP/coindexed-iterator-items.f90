! RUN: not %flang_fc1 -fopenmp -fopenmp-version=60 -fsyntax-only %s 2>&1 | FileCheck %s


module declare_mapper_iterator_coarray
  integer, save :: a(10)[*]
  type :: t
    integer :: x
  end type

  ! CHECK: A coindexed object is not allowed as a list item on MAP clause
  !$omp declare mapper(mm: t :: v) map(iterator(i = 1:10): a(i)[1])
end module


subroutine target_update_unreferenced_coindexed(a)
  integer :: a(2)
  integer, save :: ca(2)[*]

  ! CHECK: A coindexed object is not allowed as a list item on TO clause
  !$omp target update to(iterator(i = 1:2): a(i), ca(1)[1])
end subroutine
