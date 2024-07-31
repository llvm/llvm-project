!REQUIRES: amdgpu-registered-target

!RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s
!RUN: bbc -fopenmp -fopenmp-is-target-device -fopenmp-is-gpu -emit-hlfir -o - %s | FileCheck %s
!RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir -fopenmp -fopenmp-is-target-device -nogpulib %s -o - | FileCheck %s -check-prefix=FLAG_SET
!RUN: bbc -fopenmp -fopenmp-is-target-device -fopenmp-is-gpu -emit-hlfir -nogpulib -o - %s | FileCheck %s -check-prefix=FLAG_SET

!CHECK-NOT: module attributes {{{.*}}no_gpu_lib
!FLAG_SET: module attributes {{{.*}}no_gpu_lib = true
subroutine omp_subroutine()
end subroutine omp_subroutine

