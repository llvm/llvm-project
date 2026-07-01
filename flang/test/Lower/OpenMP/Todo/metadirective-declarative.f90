! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: declarative METADIRECTIVE variant

subroutine test_declarative_variant()
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: declare target) &
  !$omp & otherwise(nothing)
end subroutine

