!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!REQUIRES: amdgpu-registered-target
!RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-llvm -fopenmp -fopenmp-version=50 -fopenmp-is-target-device %s -o - | FileCheck %s

! This tests the fix for https://github.com/llvm/llvm-project/issues/84606
! We are only interested in ensuring that the -mlir-to-llmvir pass doesn't crash.

! CHECK: define weak_odr protected amdgpu_kernel void @{{.*}}QQmain{{.*}}({{.*}})
program main
  implicit none
  integer, parameter :: N = 5
  integer, dimension(5) :: a
  integer :: i
  integer :: target_a = 0

  !$omp task depend(out:a)
  do i = 1, N
    a(i) = i
  end do
  !$omp end task

  !$omp target map(tofrom:target_a) map(tofrom:a)
  do i = 1, N
    target_a = target_a + i
    a(i) = a(i) + i
  end do
  !$omp end target
  print*, target_a
  print*, a
end program main
