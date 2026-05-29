// Test for ompx_name clause code generation
//
// This test verifies that the ompx_name clause correctly sets the kernel name.
//
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s

// expected-no-diagnostics

#define TO_STR(x) #x

// CHECK: define {{.*}} @my_custom_kernel(
void test_target() {
  #pragma omp target ompx_name("my_custom_kernel")
  {
  }
}

// CHECK: define {{.*}} @another_kernel(
void test_target_parallel() {
  #pragma omp target parallel ompx_name("another_kernel")
  {
  }
}

// CHECK: define {{.*}} @teams_kernel_name(
void test_target_teams() {
  #pragma omp target teams ompx_name("teams_kernel_" "name")
  {
  }
}

// CHECK: define {{.*}} @simd_kernel_name(
void test_target_simd() {
  #pragma omp target simd ompx_name("simd_kernel_name")
  for (int i = 0; i < 10; i++)
    ;
}

// CHECK: define {{.*}} @parallel_for_kernel_3(
void test_target_parallel_for() {
  #pragma omp target parallel for ompx_name("parallel_for_kernel_" TO_STR(3))
  for (int i = 0; i < 10; i++)
    ;
}

// Verify default kernel name generation without ompx_name
// CHECK: define {{.*}} @__omp_offloading_{{[0-9a-f]+}}_{{[0-9a-f]+}}_{{.*}}_l{{[0-9]+}}(
void test_default_name() {
  #pragma omp target
  {
  }
}
