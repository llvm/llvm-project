! A DEFAULT(SHARED) clause does not require variant-local host associations and
! can be lowered with eager privatization.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -mmlir --enable-delayed-privatization=false -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_block_eager_default_shared(
! CHECK:         omp.parallel {
! CHECK:           hlfir.assign
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_block_eager_default_shared(x)
  integer :: x
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel default(shared)) &
  !$omp & otherwise(nothing)
  x = 1
  !$omp end metadirective
end subroutine
