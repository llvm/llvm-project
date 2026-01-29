!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!REQUIRES: amdgpu-registered-target
!RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-llvm -fopenmp -fopenmp-version=50 -fopenmp-is-target-device %s -o - | FileCheck %s

! The aim of this test is to verify host threadprivate directives do not cause
! crashes during OpenMP target device codegen when used in conjunction with
! target code in the same function.

! CHECK: define weak_odr protected amdgpu_kernel void @{{.*}}(ptr %{{.*}}, ptr %[[ARG1:.*]], ptr %[[ARG2:.*]]) #{{[0-9]+}} {
! CHECK:  %[[ALLOCA_X:.*]] = alloca ptr, align 8, addrspace(5)
! CHECK:  %[[ASCAST_X:.*]] = addrspacecast ptr addrspace(5) %[[ALLOCA_X]] to ptr
! CHECK:  store ptr %[[ARG1]], ptr %[[ASCAST_X]], align 8

! CHECK:  %[[ALLOCA_N:.*]] = alloca ptr, align 8, addrspace(5)
! CHECK:  %[[ASCAST_N:.*]] = addrspacecast ptr addrspace(5) %[[ALLOCA_N]] to ptr
! CHECK:  store ptr %[[ARG2]], ptr %[[ASCAST_N]], align 8

! CHECK:  %[[LOAD_X:.*]] = load ptr, ptr %[[ASCAST_X]], align 8
! CHECK:  call void @bar_(ptr %[[LOAD_X]], ptr %[[ASCAST_N]])

module test
  implicit none
  integer :: n
  !$omp threadprivate(n)
  
  contains
  subroutine foo(x)
    integer, intent(inout) :: x(10)
    !$omp target map(tofrom: x(1:n))
      call bar(x, n)
    !$omp end target
  end subroutine
end module
