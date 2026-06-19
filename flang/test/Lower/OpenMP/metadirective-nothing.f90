! Test begin metadirective with the nothing directive as a variant.
! The nothing directive as a begin-metadirective variant requires OpenMP 5.1+,
! which added it as an exception to the paired-end-directive rule.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_begin_nothing_variant()
! CHECK-NOT:     omp.parallel
! CHECK:         return
subroutine test_begin_nothing_variant()
  integer :: x
  x = 0
  !$omp begin metadirective &
#ifdef OMP_52
  !$omp & when(implementation={vendor(llvm)}: nothing) &
  !$omp & otherwise(parallel)
#else
  !$omp & when(implementation={vendor(llvm)}: nothing) &
  !$omp & default(parallel)
#endif
  x = 1
  !$omp end metadirective
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_nothing_default()
! CHECK-NOT:     omp.parallel
! CHECK:         return
subroutine test_begin_nothing_default()
  integer :: x
  x = 0
  !$omp begin metadirective &
#ifdef OMP_52
  !$omp & when(implementation={vendor("unknown")}: parallel) &
  !$omp & otherwise(nothing)
#else
  !$omp & when(implementation={vendor("unknown")}: parallel) &
  !$omp & default(nothing)
#endif
  x = 1
  !$omp end metadirective
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_nothing_first_match()
! CHECK-NOT:     omp.parallel
! CHECK-NOT:     omp.task
! CHECK:         return
subroutine test_begin_nothing_first_match()
  integer :: x
  x = 0
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: nothing) &
#ifdef OMP_52
  !$omp & when(user={condition(.false.)}: task) &
  !$omp & otherwise(parallel)
#else
  !$omp & when(user={condition(.false.)}: task) &
  !$omp & default(parallel)
#endif
  x = 1
  !$omp end metadirective
end subroutine
