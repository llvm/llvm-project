!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

! Check that this compiles successfully, but not rely on any specific output.

!CHECK: omp.parallel

program omp_cdir_crash
  implicit none
  integer, parameter :: n = 10
  real :: a(n)
  integer :: i

!$omp parallel do
!dir$ unroll
  do i = 1, n
    a(i) = real(i)
  end do
!$omp end parallel do

  print *, 'a(1)=', a(1), ' a(n)=', a(n)
end program omp_cdir_crash
