!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -fopenmp -emit-llvm %s -o - | FileCheck %s
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -emit-llvm -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s %}

subroutine lastprivate_allocatable_barrier_host
  integer, allocatable :: a
  integer :: i
  !$omp parallel do lastprivate(a)
  do i = 1, 10
    a = i
  end do
  !$omp end parallel do
end subroutine

subroutine lastprivate_allocatable_barrier_device
  integer, allocatable :: a
  integer :: i
  allocate(a)
  !$omp target parallel do lastprivate(a)
  do i = 1, 10
    a = i
  end do
  !$omp end target parallel do
end subroutine

! CHECK-LABEL: define internal void @{{.*}}lastprivate_allocatable_barrier_{{(host|device)}}
! CHECK:         call void @__kmpc_barrier
! CHECK-NEXT:    br label %omp.wsloop.region
! CHECK:         call void @__kmpc_barrier
! CHECK-NEXT:    br label %omp_loop.after
! CHECK-LABEL: define{{.*}}void @{{.*}}lastprivate_allocatable_barrier_device
