! DEFAULT(PRIVATE) still requires variant-local host associations before a
! metadirective block variant can use eager privatization.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -mmlir --enable-delayed-privatization=false -o - %s 2>&1 \
! RUN:   | FileCheck %s

! CHECK: not yet implemented: METADIRECTIVE block variant with a clause requiring variant-local host association

subroutine test_block_eager_default_private(x)
  integer :: x
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel default(private)) &
  !$omp & otherwise(nothing)
  x = 1
  !$omp end metadirective
end subroutine
