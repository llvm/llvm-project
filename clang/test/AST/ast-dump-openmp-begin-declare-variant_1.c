// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm %s -o -        | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm %s -o - -x c++ | FileCheck %s
// expected-no-diagnostics

int also_before(void) {
  return 0;
}

#pragma omp begin declare variant match(device={kind(gpu)})
int also_after(void) {
  return 1;
}
int also_before(void) {
  return 2;
}
#pragma omp end declare variant


#pragma omp begin declare variant match(device={kind(fpga)})

This text is never parsed!

#pragma omp end declare variant

int also_after(void) {
  return 0;
}

int test(void) {
  // Should return 0.
  return also_after() + also_before();
}

// CHECK: call {{.*}} @{{.*}}also_after{{[^$]*"?\(\)}}
// CHECK: call {{.*}} @{{.*}}also_before{{[^$]*"?\(\)}}
