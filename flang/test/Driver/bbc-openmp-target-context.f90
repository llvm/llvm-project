! RUN: not bbc -target x86_64-unknown-linux-gnu -fopenmp \
! RUN:   -fopenmp-version=52 -emit-hlfir %s -o /dev/null 2>&1 \
! RUN:   | FileCheck %s

! CHECK-NOT: not yet implemented
! CHECK: error: {{.*}}This construct requires a nest of depth 2, but the
! CHECK-SAME: associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2

subroutine cpu_collapse(n, a)
  integer :: n, a(n), i
  !$omp metadirective when(device={kind(cpu)}: simd collapse(2)) &
  !$omp& otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine
