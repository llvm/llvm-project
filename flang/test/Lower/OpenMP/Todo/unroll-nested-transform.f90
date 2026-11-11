! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: loop transformation nested inside an UNROLL construct

! Chaining a transformation onto the result of UNROLL needs the unrolled loop to
! be available as a generatee, which omp.unroll_* does not provide. Before this
! was diagnosed the nested construct was silently dropped.
subroutine unroll_nested_tile
  integer :: res, i
  !$omp unroll full
  !$omp tile sizes(4)
  do i = 1, 100
    res = i
  end do
  !$omp end tile
  !$omp end unroll
end subroutine
