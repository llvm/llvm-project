! Test that a metadirective resolving to a loop-associated variant
! without an associated DO loop correctly reports a TODO.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: loop-associated METADIRECTIVE without associated DO

subroutine test_no_associated_do()
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel do) &
  !$omp & default(nothing)
end subroutine
