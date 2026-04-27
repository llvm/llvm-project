! Test lowering of OpenMP metadirective with constant-folded user selectors.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_condition_true()
! CHECK:         omp.taskyield
! CHECK-NOT:     fir.if
! CHECK:         return
subroutine test_condition_true()
  !$omp metadirective &
  !$omp & when(user={condition(.true.)}: taskyield) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_condition_false()
! CHECK-NOT:     omp.taskwait
! CHECK-NOT:     fir.if
! CHECK:         return
subroutine test_condition_false()
  !$omp metadirective &
  !$omp & when(user={condition(.false.)}: taskwait) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine
