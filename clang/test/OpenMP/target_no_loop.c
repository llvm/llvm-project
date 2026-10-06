// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux-gnu \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc

// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -fopenmp-assume-teams-oversubscription \
// RUN:   -fopenmp-assume-threads-oversubscription \
// RUN:   -emit-llvm %s -o - | FileCheck %s \
// RUN:   --check-prefixes=NOLOOP

// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -emit-llvm %s -o - | FileCheck %s \
// RUN:   --check-prefix=SPMD --implicit-check-not=__kmpc_distribute_for_static_loop_4u

// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -fopenmp-assume-teams-oversubscription \
// RUN:   -emit-llvm %s -o - | FileCheck %s \
// RUN:   --check-prefix=SPMD --implicit-check-not=__kmpc_distribute_for_static_loop_4u

// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgcn-amd-amdhsa \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc \
// RUN:   -fopenmp-assume-threads-oversubscription \
// RUN:   -emit-llvm %s -o - | FileCheck %s \
// RUN:   --check-prefix=SPMD --implicit-check-not=__kmpc_distribute_for_static_loop_4u

// expected-no-diagnostics

void no_loop(int *array) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;
}

void no_loop_simd(int *array) {
#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;
}

void no_loop_nowait(int *array) {
#pragma omp target teams distribute parallel for nowait
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;
}

void spmd_collapse2(int *array) {
#pragma omp target teams distribute parallel for collapse(2)
  for (int i = 0; i < 32; ++i)
    for (int j = 0; j < 32; ++j)
      array[i * 32 + j] = i + j;
}

void spmd_schedule_static(int *array) {
#pragma omp target teams distribute parallel for schedule(static)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;
}

void spmd_schedule_guided(int *array) {
#pragma omp target teams distribute parallel for schedule(guided)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;
}

void spmd_dist_schedule(int *array) {
#pragma omp target teams distribute parallel for dist_schedule(static)
  for (int i = 0; i < 1024; ++i)
    array[i] = i + 1;
}

void spmd_cancel(int *array) {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < 1024; ++i) {
    array[i] = i + 1;
#pragma omp cancel for
  }
}

void spmd_lastprivate_scalar(int *array) {
  int last = 0;
#pragma omp target teams distribute parallel for lastprivate(last)
  for (int i = 0; i < 1024; ++i) {
    array[i] = i + 1;
    last = i;
  }
}

void spmd_linear(int *array) {
  int i;
#pragma omp target teams distribute parallel for simd linear(i)
  for (i = 0; i < 1024; ++i)
    array[i] = i + 1;
}

// NOLOOP: no_loop_l{{[0-9]+}}_kernel_environment {{.*}} i8 0, i8 1, i8 6
// NOLOOP: no_loop_simd_l{{[0-9]+}}_kernel_environment {{.*}} i8 0, i8 1, i8 6

// NOLOOP: spmd_collapse2_l{{[0-9]+}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// NOLOOP: spmd_schedule_static_l{{[0-9]+}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// NOLOOP: spmd_schedule_guided_l{{[0-9]+}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// NOLOOP: spmd_dist_schedule_l{{[0-9]+}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// NOLOOP: spmd_cancel_l{{[0-9]+}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// NOLOOP: spmd_lastprivate_scalar_l{{[0-9]+}}_kernel_environment {{.*}} i8 0, i8 1, i8 2
// NOLOOP: spmd_linear_l{{[0-9]+}}_kernel_environment {{.*}} i8 0, i8 1, i8 2

// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}no_loop_l{{[0-9]+}}{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}no_loop_l{{[0-9]+}}{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}no_loop_simd{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: store i32 1024, ptr %i
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}no_loop_simd{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// NOLOOP-LABEL: @__kmpc_parallel_60({{.*}}no_loop_nowait{{.*}})
// NOLOOP: omp.loop.exit:
// NOLOOP-NEXT: ret void
// NOLOOP: @__kmpc_distribute_for_static_loop_4u({{.*}}no_loop_nowait{{.*}}, i32 0, i32 0, i8 1)
// NOLOOP: omp_loop.exit:
// NOLOOP-NEXT: br label %omp_loop.after
// NOLOOP: omp_loop.after:
// NOLOOP-NEXT: ret void

// SPMD-COUNT-10: _kernel_environment {{.*}} i8 0, i8 1, i8 2
