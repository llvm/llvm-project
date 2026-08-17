! Test that lastprivate(conditional:) on the composite `parallel do simd`
! construct fails gracefully with a TODO rather than silently miscompiling.
! The clause reaches the do-simd composite leaf, which does not yet implement
! conditional lastprivate.

! RUN: not bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - 2>&1 | FileCheck %s

! CHECK: not yet implemented: lastprivate(conditional:) on do simd composite construct

subroutine parallel_do_simd_conditional(n, x)
  integer :: n, x, i
  x = 0
  !$omp parallel do simd lastprivate(conditional: x)
  do i = 1, n
    if (mod(i, 2) == 0) x = i
  end do
  !$omp end parallel do simd
end subroutine
