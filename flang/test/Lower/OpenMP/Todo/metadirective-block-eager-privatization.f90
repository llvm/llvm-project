! Metadirective replacements with privatizing clauses need variant-local host
! associations before they can use eager privatization.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -mmlir --enable-delayed-privatization=false -o - %s 2>&1 \
! RUN:   | FileCheck %s

! CHECK: not yet implemented: METADIRECTIVE block variant with a clause requiring variant-local host association

subroutine test_block_eager_privatization(x)
  integer :: x
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel private(x)) &
  !$omp & otherwise(nothing)
  x = 1
  !$omp end metadirective
end subroutine
