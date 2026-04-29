! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMP BEGIN/END METADIRECTIVE lowering
subroutine test_begin_metadirective
  integer :: x
  x = 0
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel) &
  !$omp & otherwise(nothing)
  x = 1
  !$omp end metadirective
end subroutine
