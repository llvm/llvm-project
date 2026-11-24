// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=52 \
// RUN:   -fopenmp-metadirective-runtime-selection -x c++ \
// RUN:   -triple x86_64-unknown-linux -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

void bar();

// With -fopenmp-simd, the body of a runtime metadirective is still emitted.
// CHECK-LABEL: define {{.*}}void @_Z3fooi(
// CHECK: call void @_Z3barv()
// CHECK-NOT: __kmpc
// CHECK: ret void
void foo(int flag) {
#pragma omp metadirective when(user={condition(flag)}: parallel) otherwise(single)
  bar();
}
