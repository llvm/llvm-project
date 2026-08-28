// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++| FileCheck %s
// expected-no-diagnostics

int also_before(void) {
  return 0;
}

#pragma omp begin declare variant match(implementation={vendor(llvm)})
int also_after(void) {
  return 1;
}
int also_before(void) {
  return 2;
}
#pragma omp end declare variant

int also_after(void) {
  return 0;
}

int test(int (*fd)(void)) {
  return fd();
}
int main(void) {
  // Should return 0.
  return test(also_after) +
         test(also_before) +
         test(&also_after) +
         test(&also_before);
}

// CHECK: call {{.*}} @{{.*}}also_after{{[^$]*"?}}
// CHECK: call {{.*}} @{{.*}}also_before{{[^$]*"?}}
// CHECK: call {{.*}} @{{.*}}also_after{{[^$]*"?}}
// CHECK: call {{.*}} @{{.*}}also_before{{[^$]*"?}}
