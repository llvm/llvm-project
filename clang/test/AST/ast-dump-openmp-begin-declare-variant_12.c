// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++| FileCheck %s
// expected-no-diagnostics

#ifdef __cplusplus
#define OVERLOADABLE
#else
#define OVERLOADABLE __attribute__((overloadable))
#endif

OVERLOADABLE
int also_before(void) {
  return 1;
}
OVERLOADABLE
int also_before(int i) {
  return 2;
}
OVERLOADABLE
int also_before(float f) {
  return 0;
}
OVERLOADABLE
int also_before(double d) {
  return 3;
}
OVERLOADABLE
int also_before(long l) {
  return 4;
}

#pragma omp begin declare variant match(implementation = {vendor(llvm)})
OVERLOADABLE
int also_before(void) {
  return 0;
}
OVERLOADABLE
int also_before(int i) {
  return 0;
}
// No float!
OVERLOADABLE
int also_before(double d) {
  return 0;
}
OVERLOADABLE
int also_before(long l) {
  return 0;
}
#pragma omp end declare variant


int main(void) {
  // Should return 0.
  return also_before() + also_before(1) + also_before(2.0f) + also_before(3.0) + also_before(4L);
}

// CHECK: call {{.*}} @{{.*}}also_before$ompvariant$S4$s11$Pllvmv
// CHECK: call {{.*}} @{{.*}}also_before$ompvariant$S4$s11$Pllvmi
// CHECK: call {{.*}} @{{.*}}also_before$ompvariant$S4$s11$Pllvmd
// CHECK: call {{.*}} @{{.*}}also_before$ompvariant$S4$s11$Pllvml
