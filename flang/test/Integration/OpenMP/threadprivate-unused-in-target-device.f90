!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!REQUIRES: amdgpu-registered-target
!RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-llvm -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s

! A procedure called from a target region may declare or use threadprivate
! variables, as long as it does not access them. Check that it can be compiled
! for the device.

! CHECK-NOT: __kmpc_threadprivate_cached
! CHECK: define {{.*}} @work_(
! CHECK-NOT: __kmpc_threadprivate_cached

module m
  integer :: mv
  !$omp threadprivate(mv)
end module

subroutine work(i)
  use m
  integer :: i
  integer, save :: y
  integer :: x
  common /c/ x
  !$omp threadprivate(/c/, y)
  i = i + 1
end subroutine

subroutine s(n)
  integer :: n
  interface
    subroutine work(i)
      integer :: i
    end subroutine
  end interface
  !$omp target
  call work(n)
  !$omp end target
end subroutine
