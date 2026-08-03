! Part 2 supports DO, SIMD, and DO SIMD loop variants only.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: loop-associated METADIRECTIVE variant other than DO, SIMD, or DO SIMD

subroutine test_loop(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: loop bind(thread)) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine
