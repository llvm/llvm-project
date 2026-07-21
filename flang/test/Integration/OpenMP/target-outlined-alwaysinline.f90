!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! On target devices the OpenMP outlined region holds the whole compute body of
! the construct. Leaving it as a separate function means it is register
! allocated without the enclosing kernel's occupancy context, which costs a
! large number of VGPRs on AMDGPU. Check that outlined regions are marked
! alwaysinline on the device so they get folded back into the kernel, and that
! the attribute is not added when the option is disabled.

! REQUIRES: amdgpu-registered-target

! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-is-target-device \
! RUN:   -triple amdgcn-amd-amdhsa -o - %s | FileCheck %s

! RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-is-target-device \
! RUN:   -triple amdgcn-amd-amdhsa \
! RUN:   -mllvm -openmp-ir-builder-device-always-inline-outlined=false \
! RUN:   -o - %s | FileCheck %s --check-prefix=DISABLED

subroutine kern(a, b, n)
  implicit none
  real(8), intent(in)    :: a(*)
  real(8), intent(inout) :: b(*)
  integer, intent(in)    :: n
  integer :: i

  !$omp target teams distribute parallel do
  do i = 1, n
     b(i) = 2.0d0 * a(i)
  end do
end subroutine kern

! The outlined loop body must carry alwaysinline on the device.
! CHECK: define internal void @{{.*}}..omp_par(i32 {{.*}}) #[[ATTR:[0-9]+]]
! CHECK: attributes #[[ATTR]] = { alwaysinline

! With the option disabled the outlined body keeps its original attributes.
! DISABLED: define internal void @{{.*}}..omp_par(i32 {{.*}}) #[[ATTR:[0-9]+]]
! DISABLED: attributes #[[ATTR]] = { "amdgpu-flat-work-group-size"
