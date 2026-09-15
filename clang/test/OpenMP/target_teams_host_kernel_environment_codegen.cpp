// Test that a '<kernel>_kernel_environment' global is generated for target
// regions compiled for a non-GPU (host-plugin) offload device, and that its
// exec mode is GENERIC and its thread/team bounds reflect the clauses used
// (or are left unset when no clause is given).

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics

void no_clauses(int n) {
  int a = 0;
#pragma omp target teams
  { a = n; }
}

void with_clauses(int n) {
  int a = 0;
#pragma omp target teams num_teams(4) thread_limit(16)
  { a = n; }
}

// CHECK: @{{.*}}no_clauses{{.*}}_kernel_environment = weak_odr protected constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 1, i32 1, i32 -1, i32 0, i32 0, i32 0 },
// CHECK: @{{.*}}with_clauses{{.*}}_kernel_environment = weak_odr protected constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 1, i32 1, i32 16, i32 4, i32 4, i32 0 },

// The host kernel does not go through the device runtime init/deinit
// sequence.
// CHECK-NOT: call {{.*}} @__kmpc_target_init(
