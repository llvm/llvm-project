!REQUIRES: amdgpu-registered-target

!RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s
!RUN: bbc -fopenmp -fopenmp-is-target-device -fopenmp-is-gpu -emit-fir -o - %s | FileCheck %s

!RUN: not %flang_fc1 -triple amdgcn-amd-amdhsa -emit-fir -fopenmp %s -o - 2>&1 | FileCheck %s --check-prefix=FLANG-ERROR
!RUN: not bbc -fopenmp -fopenmp-is-gpu -emit-fir %s -o - 2>&1 | FileCheck %s --check-prefix=BBC-ERROR

!CHECK: module attributes {{{.*}}omp.is_gpu = true
subroutine omp_subroutine()
end subroutine omp_subroutine

!FLANG-ERROR: error: OpenMP AMDGPU/NVPTX is only prepared to deal with device code.
!BBC-ERROR: FATAL: -fopenmp-is-gpu can only be set if -fopenmp-is-target-device is also set
