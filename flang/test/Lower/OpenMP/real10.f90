!REQUIRES: amdgpu-registered-target

!RUN: %flang_fc1 -emit-hlfir -fopenmp -triple amdgcn -fopenmp -fopenmp-is-target-device -o - %s | FileCheck %s

!CHECK: hlfir.declare %{{.*}} {uniq_name = "_QFEx"} : (!fir.ref<f80>) -> (!fir.ref<f80>, !fir.ref<f80>)

program p
  real(10) :: x
  !$omp target
    continue
  !$omp end target
end

