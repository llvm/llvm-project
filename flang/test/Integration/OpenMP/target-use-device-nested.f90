!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! This tests check that target code nested inside a target data region which
! has only use_device_ptr mapping correctly generates code on the device pass.

!REQUIRES: amdgpu-registered-target
!RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-llvm -fopenmp -fopenmp-version=50 -fopenmp-is-target-device %s -o - | FileCheck %s

program main
  use iso_c_binding
  implicit none
  type(c_ptr) :: a
  !$omp target data use_device_ptr(a)
    !$omp target map(tofrom: a)
      call foo(a)
    !$omp end target
  !$omp end target data
end program

! CHECK:         define weak_odr protected amdgpu_kernel void @__omp_offloading{{.*}}main_
! CHECK-NEXT:       entry:
! CHECK-NEXT:         %[[VAL_0:.*]] = call i32 @__kmpc_target_init(ptr addrspacecast (ptr addrspace(1) @__omp_offloading_{{.*}}_kernel_environment to ptr), ptr %[[VAL_6:.*]])
! CHECK-NEXT:         %[[VAL_1:.*]] = icmp eq i32 %[[VAL_0]], -1
! CHECK-NEXT:         br i1 %[[VAL_1]], label %[[USER_ENTRY:.*]], label %[[EXIT:.*]]
! CHECK:            [[USER_ENTRY]]:                                  ; preds = %entry
! CHECK-NEXT:         %[[VAL_2:.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 8) 
! CHECK-NEXT:         store ptr %[[VAL_3:.*]], ptr %[[VAL_2]], align 8
! CHECK-NEXT:         %[[VAL_4:.*]] = load ptr, ptr %[[VAL_2]], align 8
! CHECK-NEXT:         br label %[[AFTER_ALLOC:.*]]

! CHECK:            [[AFTER_ALLOC]]:
! CHECK-NEXT:         br label %[[VAL_5:.*]]

! CHECK:            [[VAL_5]]:
! CHECK-NEXT:         br label %[[TARGET_REG_ENTRY:.*]]

! CHECK:            [[TARGET_REG_ENTRY]]:                                       ; preds = %[[VAL_5]]
! CHECK-NEXT:         call void @{{.*}}foo{{.*}}(ptr %[[VAL_4]])
! CHECK-NEXT:         br label
