! Test that lastprivate(conditional:) on the generic `loop` construct fails
! gracefully with a TODO rather than silently miscompiling.  Note the `loop`
! construct does not implement lastprivate at all yet (even plain lastprivate
! hits the same TODO); this locks in the graceful-failure behavior.
! The combined `parallel loop` form behaves identically (the loop leaf hits the
! same TODO).

! RUN: not bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unhandled clause LASTPRIVATE in LOOP construct

subroutine loop_conditional(n, x)
  integer :: n, x, i
  x = 0
  !$omp loop lastprivate(conditional: x)
  do i = 1, n
    if (mod(i, 2) == 0) x = i
  end do
  !$omp end loop
end subroutine
