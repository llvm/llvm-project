! A block-associated fallback consumes the begin/end region differently from an
! ordinary loop variant and is not supported yet.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: METADIRECTIVE with both block- and loop-associated variants

subroutine test_single_fallback(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(single)
  do i = 1, n
    a(i) = i
  end do
  !$omp end metadirective
end subroutine
