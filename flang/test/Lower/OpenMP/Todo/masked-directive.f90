
! This test checks lowering of OpenMP masked Directive.

// RUN: not flang-new -fc1 -emit-fir -fopenmp %s 2>&1 | FileCheck %s

subroutine test_masked()
  integer :: c = 1
  // CHECK: not yet implemented: Unhandled Directive masked
  !$omp masked
  c = c + 1
  !$omp end masked
  // CHECK: not yet implemented: Unhandled Directive masked
  !$omp masked filter(1)
  c = c + 2
  !$omp end masked
end subroutine

subroutine test_masked_taskloop_simd()
  integer :: i, j = 1
  // CHECK: not yet implemented: Unhandled Directive masked
  !$omp masked taskloop simd
  do i=1,10
   j = j + 1
  end do
  !$omp end masked taskloop simd
end subroutine

subroutine test_masked_taskloop
  integer :: i, j = 1
  // CHECK: not yet implemented: Unhandled Directive masked
  !$omp masked taskloop filter(2)
  do i=1,10
   j = j + 1
  end do
  !$omp end masked taskloop
end subroutine

subroutine test_parallel_masked
  integer, parameter :: i = 1, j = 1
  integer :: c = 2
  // CHECK: not yet implemented: Unhandled Directive masked
  !$omp parallel masked filter(i+j)
  c = c + 2
  !$omp end parallel masked
end subroutine

subroutine test_parallel_masked_taskloop_simd
  integer :: i, j = 1
  // CHECK: not yet implemented: Unhandled Directive masked
  !$omp parallel masked taskloop simd
  do i=1,10
   j = j + 1
  end do
  !$omp end parallel masked taskloop simd
end subroutine

subroutine test_parallel_masked_taskloop
  integer :: i, j = 1
  // CHECK: not yet implemented: Unhandled Directive masked
  !$omp parallel masked taskloop filter(2)
  do i=1,10
   j = j + 1
  end do
  !$omp end parallel masked taskloop
end subroutine
