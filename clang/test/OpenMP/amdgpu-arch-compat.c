// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -fopenmp -nogpulib -fopenmp-is-target-device -verify %s
// RUN: %clang_cc1 -triple amdgpu9-amd-amdhsa -fopenmp -nogpulib -fopenmp-is-target-device -verify %s
// expected-no-diagnostics

#pragma omp begin declare variant match(device = {arch(amdgcn)})
void is_amdgcn();
#pragma omp end declare variant

// TODO: This should also accept subarch names matched for compatibility against
// the triple.
#pragma omp begin declare variant match(device = {arch(amdgpu)})
void is_amdgpu();
#pragma omp end declare variant

