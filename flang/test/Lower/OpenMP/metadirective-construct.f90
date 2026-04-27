! Test lowering of OpenMP metadirective with construct selectors.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_construct_parallel()
! CHECK:         omp.parallel
! CHECK:           omp.barrier
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_construct_parallel()
  !$omp parallel
    !$omp metadirective &
    !$omp & when(construct={parallel}: barrier) &
    !$omp & default(nothing)
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_construct_no_match()
! CHECK:         omp.parallel
! CHECK-NOT:     omp.barrier
! CHECK:         omp.taskyield
! CHECK:         omp.terminator
! CHECK:         return
subroutine test_construct_no_match()
  !$omp parallel
    !$omp metadirective &
    !$omp & when(construct={target}: barrier) &
    !$omp & default(taskyield)
  !$omp end parallel
end subroutine
