! Test that atomic capture works correctly with same element (positive control)
! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

! This test verifies that atomic capture with the same array element in both
! statements works correctly and doesn't trigger the invalid sequence diagnostic.

! CHECK-LABEL: func @_QQmain
program test_atomic_capture_same_element
  integer :: a(10)
  integer :: v

  a = 0

  ! This should work - same element a(1) in both statements
  ! CHECK: omp.atomic.capture
  !$omp atomic capture
  v = a(1)
  a(1) = a(1) + 1
  !$omp end atomic

end program test_atomic_capture_same_element
