// Test that a '<kernel>_kernel_environment' global is generated for
// 'ompx_bare' kernels, that its exec mode is BARE, that its thread/team
// bounds reflect the required 'num_teams'/'thread_limit' clauses, and that
// the bare kernel does not go through the device runtime init/deinit
// sequence.

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

template <typename tx>
tx ftemplate(int n) {
  tx a = 0;
#pragma omp target teams ompx_bare num_teams(1) thread_limit(32)
  { a = 2; }
  return a;
}

int bar(int n) {
  int a = 0;
  a += ftemplate<char>(n);
  return a;
}

// CHECK: @{{.*}}ftemplate{{.*}}_kernel_environment = weak_odr protected constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 0, i32 1, i32 32, i32 1, i32 1, i32 0 },

// The bare kernel does not go through the device runtime init/deinit
// sequence.
// CHECK-NOT: call {{.*}} @__kmpc_target_init(
