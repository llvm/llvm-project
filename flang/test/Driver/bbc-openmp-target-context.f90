! REQUIRES: x86-registered-target

! RUN: bbc -target x86_64-unknown-linux-gnu -fopenmp -fopenmp-version=52 \
! RUN:   -emit-hlfir %s -o - 2>&1 | FileCheck %s

! Semantics must use the bbc target: arch(nvptx64) does not match x86-64, so
! the invalid SIMD replacement is never selected and must not be diagnosed.
! CHECK-NOT: error
! CHECK-LABEL: func.func @_QParch_mismatch(
! CHECK-NOT: omp.simd
! CHECK: fir.do_loop

subroutine arch_mismatch(n, a)
  integer :: n, a(n), i
  !$omp metadirective when(device={arch(nvptx64)}: simd collapse(2)) &
  !$omp& otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine
