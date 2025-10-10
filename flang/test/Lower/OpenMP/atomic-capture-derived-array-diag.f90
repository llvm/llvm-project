! Test diagnostic for atomic capture with invalid operation sequence
! RUN: not %flang_fc1 -emit-fir -fopenmp %s 2>&1 | FileCheck %s

! This test verifies that a clear diagnostic is emitted when atomic capture
! produces an invalid operation sequence. This can occur with derived-type
! component array elements where different indices are used.

program test_atomic_capture_invalid_sequence
  type t1
    integer :: i(1)
  end type
  type(t1) :: t2
  integer :: j

  t2%i = 0
  j = 1

  ! CHECK: error: OpenMP atomic capture produced an invalid operation sequence
  !$omp atomic capture
  t2%i(j*1) = t2%i(1) + 1
  t2%i(1)   = t2%i(j*1)
  !$omp end atomic

end program test_atomic_capture_invalid_sequence
