! Test that atomic capture with derived-type array elements fails appropriately
! RUN: not %flang_fc1 -emit-fir -fopenmp %s 2>&1 | FileCheck %s

! This test verifies that atomic capture with derived-type component array
! elements using different indices fails during MLIR verification. The issue
! is that different array elements represent different memory locations, which
! violates OpenMP's requirement that both statements in atomic capture must
! operate on the same variable. This will fail in the MLIR verifier until
! semantic analysis is fixed to detect this earlier.

program test_atomic_capture_invalid_sequence
  type t1
    integer :: i(1)
  end type
  type(t1) :: t2
  integer :: j

  t2%i = 0
  j = 1

  ! CHECK: invalid sequence of operations in the capture region
  !$omp atomic capture
  t2%i(j*1) = t2%i(1) + 1
  t2%i(1)   = t2%i(j*1)
  !$omp end atomic

end program test_atomic_capture_invalid_sequence
