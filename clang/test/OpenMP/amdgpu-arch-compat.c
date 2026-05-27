// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -nogpulib -fopenmp-is-target-device -verify %s
// expected-no-diagnostics

#pragma omp begin declare variant match(device = {arch(amdgcn)})
void is_amdgcn();
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {arch(amdgpu)})
void is_amdgpu();
#pragma omp end declare variant


