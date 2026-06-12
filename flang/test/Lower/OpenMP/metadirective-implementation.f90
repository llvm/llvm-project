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

! CHECK-LABEL: func.func @_QPtest_implicit_nothing_tie_break()
! CHECK:         omp.barrier
! CHECK:         return
subroutine test_implicit_nothing_tie_break()
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}:) &
  !$omp & when(implementation={vendor(llvm)}: barrier)
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_vendor_llvm()
! CHECK:         omp.parallel
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_begin_vendor_llvm()
  integer :: x
  x = 0
#ifdef OMP_52
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel) &
  !$omp & otherwise(nothing)
#else
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel)
#endif
  x = 1
  !$omp end metadirective
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_vendor_no_match()
! CHECK-NOT:     omp.parallel
! CHECK:         return
subroutine test_begin_vendor_no_match()
  integer :: x
  x = 0
#ifdef OMP_52
  !$omp begin metadirective &
  !$omp & when(implementation={vendor("unknown")}: parallel) &
  !$omp & otherwise(nothing)
#else
  !$omp begin metadirective &
  !$omp & when(implementation={vendor("unknown")}: parallel)
#endif
  x = 1
  !$omp end metadirective
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_multiple_when_first_match()
! CHECK:         omp.parallel
! CHECK:           omp.terminator
! CHECK-NOT:     omp.task
! CHECK:         return
subroutine test_begin_multiple_when_first_match()
  integer :: x
  x = 0
#ifdef OMP_52
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel) &
  !$omp & when(user={condition(.false.)}: task) &
  !$omp & otherwise(nothing)
#else
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel) &
  !$omp & when(user={condition(.false.)}: task)
#endif
  x = 1
  !$omp end metadirective
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_implicit_nothing_tie_break()
! CHECK:         omp.parallel
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_begin_implicit_nothing_tie_break()
  integer :: x
  x = 0
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}:) &
  !$omp & when(implementation={vendor(llvm)}: parallel)
  x = 1
  !$omp end metadirective
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_multiple_when_fallback()
! CHECK-NOT:     omp.task
! CHECK:         omp.parallel
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_begin_multiple_when_fallback()
  integer :: x
  x = 0
  !$omp begin metadirective &
  !$omp & when(implementation={vendor("nvidia")}: task) &
  !$omp & when(user={condition(.false.)}: task) &
#ifdef OMP_52
  !$omp & otherwise(parallel)
#else
  !$omp & default(parallel)
#endif
  x = 1
  !$omp end metadirective
end subroutine
