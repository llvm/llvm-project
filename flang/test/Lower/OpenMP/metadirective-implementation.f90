! Test lowering of OpenMP metadirective with implementation selectors.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=51 %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 -cpp -DOMP_52 %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_vendor_llvm()
! CHECK:         omp.taskwait
! CHECK:         return
subroutine test_vendor_llvm()
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: taskwait) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_vendor_no_match()
! CHECK-NOT:     omp.taskwait
! CHECK:         return
subroutine test_vendor_no_match()
  !$omp metadirective &
  !$omp & when(implementation={vendor("unknown")}: taskwait) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_standalone_barrier_match()
! CHECK:         omp.barrier
! CHECK:         return
subroutine test_standalone_barrier_match()
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: barrier) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_standalone_barrier_fallback()
! CHECK:         omp.barrier
! CHECK:         return
subroutine test_standalone_barrier_fallback()
  !$omp metadirective &
  !$omp & when(implementation={vendor("cray")}: nothing) &
#ifdef OMP_52
  !$omp & otherwise(barrier)
#else
  !$omp & default(barrier)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_nothing_variant()
! CHECK-NOT:     omp.taskwait
! CHECK:         return
subroutine test_nothing_variant()
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: nothing) &
#ifdef OMP_52
  !$omp & otherwise(taskwait)
#else
  !$omp & default(taskwait)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_default_fallback()
! CHECK:         omp.taskwait
! CHECK:         return
subroutine test_default_fallback()
  !$omp metadirective &
  !$omp & when(implementation={vendor("unknown")}: nothing) &
#ifdef OMP_52
  !$omp & otherwise(taskwait)
#else
  !$omp & default(taskwait)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_no_default()
! CHECK-NOT:     omp.taskyield
! CHECK:         return
subroutine test_no_default()
  !$omp metadirective &
  !$omp & when(implementation={vendor("gnu")}: taskyield)
end subroutine

! CHECK-LABEL: func.func @_QPtest_multiple_when_first_match()
! CHECK:         omp.taskwait
! CHECK-NOT:     omp.taskyield
! CHECK:         return
subroutine test_multiple_when_first_match()
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: taskwait) &
  !$omp & when(user={condition(.false.)}: taskyield) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
end subroutine

! CHECK-LABEL: func.func @_QPtest_multiple_when_fallback()
! CHECK-NOT:     omp.taskyield
! CHECK:         omp.taskwait
! CHECK:         return
subroutine test_multiple_when_fallback()
  !$omp metadirective &
  !$omp & when(implementation={vendor("nvidia")}: taskyield) &
  !$omp & when(user={condition(.false.)}: taskyield) &
#ifdef OMP_52
  !$omp & otherwise(taskwait)
#else
  !$omp & default(taskwait)
#endif
end subroutine
