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

! CHECK-LABEL: func.func @_QPtest_begin_construct_parallel()
! CHECK:         omp.parallel
! CHECK:           omp.parallel
! CHECK:             omp.terminator
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_begin_construct_parallel()
  integer :: x
  x = 0
  !$omp parallel
    !$omp begin metadirective &
    !$omp & when(construct={parallel}: parallel)
    x = 1
    !$omp end metadirective
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_construct_no_match()
! CHECK:         omp.parallel
! CHECK-NOT:     omp.task
! CHECK:         omp.terminator
! CHECK:         return
subroutine test_begin_construct_no_match()
  integer :: x
  x = 0
  !$omp parallel
    !$omp begin metadirective &
    !$omp & when(construct={target}: task)
    x = 1
    !$omp end metadirective
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_construct_selected_parent()
! CHECK:         omp.target
! CHECK:           omp.parallel
! CHECK:             omp.barrier
! CHECK-NOT:         omp.taskyield
! CHECK:             omp.terminator
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_begin_construct_selected_parent()
  !$omp target
    !$omp begin metadirective &
    !$omp & when(implementation={vendor(llvm)}: parallel)
      !$omp metadirective &
      !$omp & when(construct={target, parallel}: barrier) &
      !$omp & default(taskyield)
    !$omp end metadirective
  !$omp end target
end subroutine
