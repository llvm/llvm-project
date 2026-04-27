!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

! Check that this compiles successfully.

!CHECK: omp.atomic.capture
!CHECK: omp.atomic.read
!CHECK: omp.atomic.update
subroutine f00
  implicit none
  real :: c
  complex, allocatable :: x
  !$omp atomic update capture
    c = x%re
    x%re = x%re + 1.0
  !$omp end atomic
end

