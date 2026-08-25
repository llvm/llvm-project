// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++| FileCheck %s
// expected-no-diagnostics
// FIXME: We have to improve the warnings here as nothing is impacted by the declare variant.
int also_before(void) {
  return 0;
}

#pragma omp begin declare variant match(device={kind(cpu)})
int also_before(void);
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score(100):llvm)})
int also_after(void);
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score(0):llvm)})
int also_before(void);
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
