// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s       | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++| FileCheck %s
// expected-no-diagnostics

int also_before(void) {
  return 1;
}

#pragma omp begin declare variant match(implementation={vendor(llvm)})
int also_after(void) {
  return 0;
}
int also_before(void) {
  return 0;
}
#pragma omp end declare variant

int also_after(void) {
  return 2;
}

int main(void) {
  // Should return 0.
  return (also_after)() +
         (also_before)() +
         (&also_after)() +
         (&also_before)();
}

// CHECK: call {{.*}} @"{{.*}}also_after$ompvariant$S4$s11$Pllvm
// CHECK: call {{.*}} @"{{.*}}also_before$ompvariant$S4$s11$Pllvm
// CHECK: call {{.*}} @"{{.*}}also_after$ompvariant$S4$s11$Pllvm
// CHECK: call {{.*}} @"{{.*}}also_before$ompvariant$S4$s11$Pllvm
