// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++| FileCheck %s
// expected-no-diagnostics

#pragma omp begin declare variant match(device={kind(cpu)})
int also_before(void) {
  return 1;
}
#pragma omp end declare variant

#pragma omp begin declare variant match(implementation={vendor(score(0):llvm)})
int also_after(void) {
  return 0;
}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score(100):llvm)})
int also_before(void) {
  return 0;
}
#pragma omp end declare variant

int also_after(void) {
  return 2;
}

int test(void) {
  // Should return 0.
  return also_after() + also_before();
}

// CHECK: call {{.*}} @{{.*}}also_after$ompvariant$S4$s11$Pllvm
// CHECK: call {{.*}} @{{.*}}also_before$ompvariant$S4$s11$Pllvm
