! RUN: %not_todo_cmd %flang_fc1 -cpp -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=60 -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -cpp -DFALLBACK -emit-hlfir -fopenmp \
! RUN:   -fopenmp-version=60 -o - %s 2>&1 | FileCheck %s

! NOTHING with APPLY must not be silently lowered as a no-op, whether it is
! selected by a WHEN clause or as the fallback.
! CHECK: not yet implemented: NOTHING with APPLY in METADIRECTIVE

subroutine nothing_apply(n, a)
  integer :: n, a(n), i
#ifdef FALLBACK
  !$omp metadirective when(user={condition(.false.)}: nothing) &
  !$omp& otherwise(nothing apply(reverse))
#else
  !$omp metadirective &
  !$omp& when(user={condition(.true.)}: nothing apply(reverse)) &
  !$omp& otherwise(nothing)
#endif
  do i = 1, n
    a(i) = i
  end do
end subroutine
