// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 \
// RUN: -triple x86_64-unknown-linux-gnu -ast-print %s -o - | FileCheck %s

// expected-no-diagnostics

void bar();
void baz();

void test_nonconstant_condition(bool use_gpu) {
  #pragma omp metadirective \
      when(user={condition(use_gpu)}: parallel) \
      otherwise(single)
  {
    bar();
  }
}
// CHECK-LABEL: void test_nonconstant_condition(bool use_gpu)
// CHECK: #pragma omp metadirective when(use_gpu: #pragma omp parallel) otherwise( #pragma omp single)
// CHECK: bar();


void test_no_otherwise(bool flag) {
  #pragma omp metadirective \
      when(user={condition(flag)}: parallel)
  {
    bar();
  }
}
// CHECK-LABEL: void test_no_otherwise(bool flag)
// CHECK: #pragma omp metadirective when(flag: #pragma omp parallel)
// CHECK: bar();


void test_parallel_for(bool use_gpu, int n) {
  #pragma omp metadirective \
      when(user={condition(use_gpu)}: parallel for) \
      otherwise(single)
  for (int i = 0; i < 10; i++)
    bar();
}
// CHECK-LABEL: void test_parallel_for(bool use_gpu, int n)
// CHECK: #pragma omp metadirective when(use_gpu: #pragma omp parallel for) otherwise( #pragma omp single)
// CHECK: bar();


void test_multiple_when(bool use_gpu, bool use_cpu) {
  #pragma omp metadirective \
      when(user={condition(use_gpu)}: parallel) \
      when(user={condition(use_cpu)}: single) \
      otherwise(simd)
  {
  for (int i = 0; i < 10; i++)
    bar();
  }
}
// CHECK-LABEL: void test_multiple_when(bool use_gpu, bool use_cpu)
// CHECK: #pragma omp metadirective when(use_gpu: #pragma omp parallel) when(use_cpu: #pragma omp single) otherwise( #pragma omp simd)
// CHECK: bar();


void test_otherwise_empty(bool flag) {
  #pragma omp metadirective \
      when(user={condition(flag)}: parallel) \
      otherwise()
  {
    bar();
  }
}
// CHECK-LABEL: void test_otherwise_empty(bool flag)
// CHECK: #pragma omp metadirective when(flag: #pragma omp parallel) otherwise()
// CHECK: bar();


bool is_gpu_available();
void test_function_condition() {
  #pragma omp metadirective \
      when(user={condition(is_gpu_available())}: parallel) \
      otherwise(single)
  {
    bar();
  }
}
// CHECK-LABEL: void test_function_condition()
// CHECK: #pragma omp metadirective when(is_gpu_available(): #pragma omp parallel) otherwise( #pragma omp single)
// CHECK: bar();


void test_mixed_conditions(bool use_gpu) {
  #pragma omp metadirective \
      when(user={condition(use_gpu)}: parallel) \
      otherwise(parallel for)
  for (int i = 0; i < 100; i++)
    bar();
}
// CHECK-LABEL: void test_mixed_conditions(bool use_gpu)
// CHECK: #pragma omp metadirective when(use_gpu: #pragma omp parallel) otherwise( #pragma omp parallel for)
// CHECK: bar();


void test_compound_body(bool flag) {
  #pragma omp metadirective \
      when(user={condition(flag)}: parallel) \
      otherwise(single)
  {
    bar();
    baz();
  }
}
// CHECK-LABEL: void test_compound_body(bool flag)
// CHECK: #pragma omp metadirective when(flag: #pragma omp parallel) otherwise( #pragma omp single)
// CHECK: bar();
// CHECK-NEXT: baz();
