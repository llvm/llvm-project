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

void foo(void);
typedef int(*fd)(void);
int main(void) {
  // Should return 0.
  fd fns[2];
  fns[0] = &also_before;
  fns[1] = also_after;
  return (foo(), also_after)() +
         (fns[0])() +
         (1[fns])();
}

// CHECK: store ptr @{{.*}}also_before{{[^$]*"?}}
// CHECK: store ptr @{{.*}}also_after{{[^$]*"?}}
// CHECK: call {{.*}} @{{.*}}also_after{{[^$]*"?\(\)}}
