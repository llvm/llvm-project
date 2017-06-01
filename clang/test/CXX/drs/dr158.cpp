// RUN: %clang_cc1 -triple x86_64-linux -std=c++98 %s -O3 -disable-llvm-passes -pedantic-errors -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c++11 %s -O3 -disable-llvm-passes -pedantic-errors -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c++14 %s -O3 -disable-llvm-passes -pedantic-errors -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c++1z %s -O3 -disable-llvm-passes -pedantic-errors -emit-llvm -o - | FileCheck %s

// dr158: yes

// CHECK-LABEL: define {{.*}} @_Z1f
const int *f(const int * const *p, int **q) {
  // CHECK: load i32**, {{.*}}, !tbaa ![[INTPTR_TBAA:[^,]*]]
  const int *x = *p;
  // CHECK: store i32* null, {{.*}}, !tbaa ![[INTPTR_TBAA]]
  *q = 0;
  return x;
}

struct A {};

// CHECK-LABEL: define {{.*}} @_Z1g
const int *(A::*const *g(const int *(A::* const **p)[3], int *(A::***q)[3]))[3] {
  // CHECK: load i64**, {{.*}}, !tbaa ![[MEMPTR_TBAA:[^,]*]]
  const int *(A::*const *x)[3] = *p;
  // CHECK: store i64* null, {{.*}}, !tbaa ![[MEMPTR_TBAA]]
  *q = 0;
  return x;
}

