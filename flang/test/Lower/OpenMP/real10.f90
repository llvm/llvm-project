!REQUIRES: amdgpu-registered-target

!RUN: %flang_fc1 -emit-hlfir -fopenmp -triple amdgcn -fopenmp -fopenmp-is-target-device -o - %s | FileCheck %s

!CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFtest_real10Ex"} : (!fir.ref<f80>) -> (!fir.ref<f80>, !fir.ref<f80>)

subroutine test_real10()
  !$omp declare target
  real(10) :: x
end subroutine test_real10
