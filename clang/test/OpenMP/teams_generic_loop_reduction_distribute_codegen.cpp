// REQUIRES: amdgpu-registered-target

// Check that a 'teams loop' with a reduction that is emitted as 'distribute'
// (rather than 'distribute parallel for') keeps its distribute loop on the
// GPU. The fused distribute schedule omits the outer distribute loop and relies
// on an inner worksharing loop, which only exists for 'distribute parallel for'.

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgpu-amd-amdhsa -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s

// expected-no-diagnostics

#pragma omp declare target
double getY(const double *y, int i) { return y[i]; }
#pragma omp end declare target

// The call to getY prevents 'target teams loop' from being emitted as
// 'distribute parallel for'.
double target_teams_loop_call(const double *y, int N) {
  double check = 0.0;
#pragma omp target teams loop reduction(+:check) map(to: y[0:N])
  for (int i = 0; i < N; i++)
    check += getY(y, i);
  return check;
}

// 'teams loop' is always emitted as 'distribute'.
double teams_loop(const double *y, int N) {
  double check = 0.0;
#pragma omp target map(to: y[0:N]) map(tofrom: check)
#pragma omp teams loop reduction(+:check)
  for (int i = 0; i < N; i++)
    check += y[i];
  return check;
}

// Emitted as 'distribute parallel for', so the fused schedule is used.
double target_teams_loop_parallel_for(const double *y, int N) {
  double check = 0.0;
#pragma omp target teams loop reduction(+:check) map(to: y[0:N])
  for (int i = 0; i < N; i++)
    check += y[i];
  return check;
}

// CHECK-LABEL: define internal void @{{.*}}target_teams_loop_call{{.*}}_l{{[0-9]+}}_omp_outlined(
// CHECK: call void @__kmpc_distribute_static_init_4(
// CHECK: omp.inner.for.body:
// CHECK: call noundef double @_Z4getYPKdi(
// CHECK: call void @__kmpc_distribute_static_fini(
// CHECK: call i32 @__kmpc_gpu_xteam_reduce_nowait(

// CHECK-LABEL: define internal void @{{.*}}teams_loop{{.*}}_l{{[0-9]+}}_omp_outlined(
// CHECK: call void @__kmpc_distribute_static_init_4(
// CHECK: omp.inner.for.body:
// CHECK: call void @__kmpc_distribute_static_fini(
// CHECK: call i32 @__kmpc_gpu_xteam_reduce_nowait(

// CHECK-LABEL: define internal void @{{.*}}target_teams_loop_parallel_for{{.*}}_l{{[0-9]+}}_omp_outlined(
// CHECK-NOT: call void @__kmpc_distribute_static_init
// CHECK: call void @__kmpc_parallel_60(
// CHECK-LABEL: define internal void @{{.*}}target_teams_loop_parallel_for{{.*}}_l{{[0-9]+}}_omp_outlined_omp_outlined(
// CHECK: call void @__kmpc_for_static_init_4(ptr {{.*}}, i32 {{.*}}, i32 93,
