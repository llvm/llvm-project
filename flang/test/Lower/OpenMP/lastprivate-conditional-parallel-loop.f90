! Test that lastprivate(conditional:) on the composite `parallel loop` construct
! fails gracefully with a TODO rather than silently miscompiling.

! RUN: not bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unhandled clause LASTPRIVATE in LOOP construct

subroutine parallel_loop_conditional(n, x)
  integer :: n, x, i
  x = 0
  !$omp parallel loop lastprivate(conditional: x)
  do i = 1, n
    if (mod(i, 2) == 0) x = i
  end do
  !$omp end parallel loop
end subroutine
