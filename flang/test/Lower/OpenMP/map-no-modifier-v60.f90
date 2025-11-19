!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

!This shouldn't crash. Check for a symptom of a successful compilation
!CHECK: omp.map.info

subroutine f00
  implicit none
  integer :: x
  !$omp target map(x)
  !$omp end target
end
  
