!REQUIRES: amdgpu-registered-target

!RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s
!RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s
!RUN: bbc -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s 
!RUN: bbc -emit-fir -fopenmp -fopenmp-is-gpu -fopenmp-is-target-device %s -o - | FileCheck %s 

!CHECK: func.func @_QPtarget_function

!CHECK:  func.func @_QPwrite_index_omp_outline_0(%[[ARG0:.*]]: !fir.ref<i32>) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QPwrite_index"} {
!CHECK-NEXT: omp.target  {{.*}} {
!CHECK: %[[CONSTANT_VALUE_10:.*]] = arith.constant 10 : i32
!CHECK: fir.store %[[CONSTANT_VALUE_10]] to %[[ARG0]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK-NEXT: }
!CHECK-NEXT: return

!CHECK:  func.func @_QPwrite_index_omp_outline_1(%[[ARG1:.*]]: !fir.ref<i32>) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QPwrite_index"} {
!CHECK-NEXT: omp.target  {{.*}} {
!CHECK: %[[CONSTANT_VALUE_20:.*]] = arith.constant 20 : i32
!CHECK: fir.store %[[CONSTANT_VALUE_20]] to %[[ARG1]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK-NEXT: }
!CHECK-NEXT: return


SUBROUTINE WRITE_INDEX(INT_ARRAY)
        INTEGER :: INT_ARRAY(*)
        INTEGER :: NEW_LEN
!$omp target map(from:new_len)
        NEW_LEN = 10
!$omp end target
!$omp target map(from:new_len)
        NEW_LEN = 20
!$omp end target
        do INDEX_ = 1, NEW_LEN
                INT_ARRAY(INDEX_) = INDEX_
        end do
end subroutine WRITE_INDEX

SUBROUTINE TARGET_FUNCTION()
!$omp declare target
END
