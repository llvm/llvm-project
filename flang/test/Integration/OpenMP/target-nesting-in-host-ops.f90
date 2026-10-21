!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!REQUIRES: amdgpu-registered-target
!RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-llvm -fopenmp -fopenmp-version=50 -fopenmp-is-target-device %s -o - | FileCheck %s

! CHECK-NOT: define void @nested_target_in_parallel
! CHECK: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_nested_target_in_parallel_{{.*}}(ptr %{{.*}}, ptr %{{.*}})
subroutine nested_target_in_parallel(v)
  implicit none
  integer, intent(inout) :: v(10)

  !$omp parallel
    !$omp target map(tofrom: v)
    !$omp end target
  !$omp end parallel
end subroutine

! CHECK-NOT: define void @nested_target_in_wsloop
! CHECK: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_nested_target_in_wsloop_{{.*}}(ptr %{{.*}}, ptr %{{.*}})
subroutine nested_target_in_wsloop(v)
  implicit none
  integer, intent(inout) :: v(10)
  integer :: i

  !$omp do
  do i=1, 10
    !$omp target map(tofrom: v)
    !$omp end target
  end do
end subroutine

! CHECK-NOT: define void @nested_target_in_parallel_with_private
! CHECK: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_nested_target_in_parallel_with_private_{{.*}}(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
subroutine nested_target_in_parallel_with_private(v)
  implicit none
  integer, intent(inout) :: v(10)
  integer :: x
  x = 10

  !$omp parallel firstprivate(x)
    !$omp target map(tofrom: v(1:x))
    !$omp end target
  !$omp end parallel
end subroutine

! CHECK-NOT: define void @nested_target_in_task_with_private
! CHECK: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_nested_target_in_task_with_private_{{.*}}(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
subroutine nested_target_in_task_with_private(v)
  implicit none
  integer, intent(inout) :: v(10)
  integer :: x
  x = 10

  !$omp task firstprivate(x)
    !$omp target map(tofrom: v(1:x))
    !$omp end target
  !$omp end task
end subroutine

! CHECK-NOT: define void @target_and_atomic_update
! CHECK: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_and_atomic_update_{{.*}}(ptr %{{.*}})
subroutine target_and_atomic_update(x, expr)
  implicit none
  integer, intent(inout) :: x, expr

  !$omp target
  !$omp end target

  !$omp atomic update
  x = x + expr
end subroutine

! CHECK-NOT: define void @nested_target_in_associate
! CHECK: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_nested_target_in_associate_{{.*}}(ptr %{{.*}}, ptr %{{.*}}, ptr %{{.*}})
subroutine nested_target_in_associate(x)
  integer, pointer, contiguous :: x(:)
  associate(y => x)
    !$omp target map(tofrom: y)
    !$omp end target
  end associate
end subroutine
