// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++| FileCheck %s
// expected-no-diagnostics

#ifdef __cplusplus
#define CONST constexpr
#else
#define CONST __attribute__((const))
#endif

int also_before1(void) {
  return 1;
}
int also_before2(void) {
  return 2;
}
int also_before3(void) {
  return 3;
}
int also_before4(void) {
  return 4;
}

#pragma omp begin declare variant match(implementation = {vendor(llvm)})
CONST int also_before1(void) {
  return 0;
}
static int also_before2(void) {
  return 0;
}
__attribute__((nothrow)) int also_before3(void) {
  return 0;
}
static CONST __attribute__((nothrow, always_inline)) __inline__ int also_before4(void) {
  return 0;
}
#pragma omp end declare variant


int main(void) {
  // Should return 0.
  return also_before1() + also_before2() + also_before3() + also_before4();
}

// CHECK:     call {{.*}} @{{.*}}also_before1$ompvariant$S4$s11$Pllvm
// CHECK:     call {{.*}} @{{.*}}also_before2$ompvariant$S4$s11$Pllvm
// CHECK:     call {{.*}} @{{.*}}also_before3$ompvariant$S4$s11$Pllvm
// CHECK-NOT: call {{.*}} @{{.*}}also_before4{{[^$]*"?\(\)}}
