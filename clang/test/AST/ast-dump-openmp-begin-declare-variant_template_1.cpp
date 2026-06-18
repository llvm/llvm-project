// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++ | FileCheck %s
// expected-no-diagnostics

int also_before() {
  return 1;
}

#pragma omp begin declare variant match(implementation={vendor(score(100):llvm)})
int also_after(void) {
  return 2;
}
int also_after(int) {
  return 3;
}
int also_after(double) {
  return 0;
}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score(0):llvm)})
int also_before() {
  return 0;
}
#pragma omp end declare variant

int also_after(void) {
  return 4;
}
int also_after(int) {
  return 5;
}
int also_after(double) {
  return 6;
}

template<typename T>
int test1() {
  // Should return 0.
  return also_after(T(0));
}

typedef int(*Ty)();

template<Ty fn>
int test2() {
  // Should return 0.
  return fn();
}

int test() {
  // Should return 0.
  return test1<double>() + test2<also_before>();
}

// CHECK: call {{.*}} @"_Z34also_after$ompvariant$S4$s11$Pllvmd"
// CHECK: call {{.*}} @"_Z35also_before$ompvariant$S4$s11$Pllvmv"
