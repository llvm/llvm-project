// RUN: %clang_cc1 -verify -fopenmp -fblocks -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fblocks -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=SIMD-ONLY
// expected-no-diagnostics

// Target regions inside lambdas and blocks at namespace scope have no parent
// function declaration, so they are named after the function being emitted.

int x;
int &ref = x;

auto lambda = []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+1]]{{"?}}(
#pragma omp target
  ref = 42;
};

auto nested_lambda = []() {
  return []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+1]]{{"?}}(
#pragma omp target
    ref = 42;
  };
};

auto combined_directive = []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+1]]{{"?}}(
#pragma omp target teams
  ref = 42;
};

auto static_local = []() {
  static int &local_ref = x;
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+1]]{{"?}}(
#pragma omp target
  local_ref = 42;
};

void (^block)() = ^{
// CHECK-DAG: define internal void @__omp_offloading_{{.*}}_block_block_invoke_l[[#@LINE+1]](
#pragma omp target
  ref = 42;
};

void default_argument(int = []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+1]]{{"?}}(
#pragma omp target
  ref = 42;
  return 0;
}());

template <int N> int variable_template = []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+1]]{{"?}}(
#pragma omp target
  ref = N;
  return 0;
}();
int instantiation = variable_template<1>;

void use() {
  lambda();
  nested_lambda()();
  combined_directive();
  static_local();
  default_argument();
}

// SIMD-ONLY-NOT: {{__kmpc|__tgt}}
