// REQUIRES: amdgpu-registered-target

// A 'teams loop' nested in a 'target' region is emitted as 'teams distribute',
// so the kernel must use generic mode, unlike a nested 'teams distribute
// parallel for'.

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgpu-amd-amdhsa -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s

// expected-no-diagnostics

// The exec mode is the third field of the configuration environment: generic
// is 1, SPMD is 2.
// CHECK: @{{.*}}teams_loop{{.*}}_kernel_environment = {{.*}} { %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 1,
// CHECK: @{{.*}}teams_loop_no_reduction{{.*}}_kernel_environment = {{.*}} { %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 1,
// CHECK: @{{.*}}teams_distribute_parallel_for{{.*}}_kernel_environment = {{.*}} { %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 2,

void teams_loop(double *y, double &c, int N) {
#pragma omp target map(to: y[0:N]) map(tofrom: c)
#pragma omp teams loop reduction(+:c)
  for (int i = 0; i < N; i++)
    c += y[i];
}

void teams_loop_no_reduction(int *z, int N) {
#pragma omp target map(tofrom: z[0:N])
#pragma omp teams loop
  for (int i = 0; i < N; i++)
    z[i] += 1;
}

void teams_distribute_parallel_for(int *z, int N) {
#pragma omp target map(tofrom: z[0:N])
#pragma omp teams distribute parallel for
  for (int i = 0; i < N; i++)
    z[i] += 1;
}
