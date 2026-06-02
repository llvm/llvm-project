! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: loop-associated METADIRECTIVE variant

subroutine test_loop_variant()
  integer :: i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel do) &
  !$omp & default(nothing)
  do i = 1, 100
  end do
end subroutine
