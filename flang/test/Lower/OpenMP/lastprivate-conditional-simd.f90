! Test that lastprivate(conditional:) on a standalone simd construct
! produces a TODO diagnostic.

! RUN: not bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - 2>&1 | FileCheck %s
! RUN: not %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - 2>&1 | FileCheck %s

! CHECK: not yet implemented: lastprivate(conditional:) on simd construct

subroutine simd_conditional(a, n)
  integer :: a(:), n, x, i
  x = 0
  !$omp simd lastprivate(conditional: x)
  do i = 1, n
    if (a(i) > 0) x = a(i)
  end do
  !$omp end simd
  a(1) = x
end subroutine
