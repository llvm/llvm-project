! Data-sharing clauses on selected loop variants need their symbol attributes
! reconstructed during lowering.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: data-sharing clause in loop-associated METADIRECTIVE variant

subroutine test_private(n, a)
  integer :: n, a(n), i, x
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do private(x)) &
  !$omp & otherwise(nothing)
  do i = 1, n
    x = i
    a(i) = x
  end do
end subroutine
