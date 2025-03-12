// RUN: %clang_cc1 -triple x86_64-linux -std=c++98 %s -O3 -disable-llvm-passes -pedantic-errors -emit-llvm -o - | FileCheck --check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c++11 %s -O3 -disable-llvm-passes -pedantic-errors -emit-llvm -o - | FileCheck --check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c++14 %s -O3 -disable-llvm-passes -pedantic-errors -emit-llvm -o - | FileCheck --check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c++17 %s -O3 -disable-llvm-passes -pedantic-errors -emit-llvm -o - | FileCheck --check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c++20 %s -O3 -disable-llvm-passes -pedantic-errors -emit-llvm -o - | FileCheck --check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c++23 %s -O3 -disable-llvm-passes -pedantic-errors -emit-llvm -o - | FileCheck --check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c++2c %s -O3 -disable-llvm-passes -pedantic-errors -emit-llvm -o - | FileCheck --check-prefixes=CHECK %s
// RUN: %clang_cc1 -triple x86_64-linux -std=c++1z %s -O3 -pointer-tbaa -disable-llvm-passes -pedantic-errors -emit-llvm -o - | FileCheck --check-prefixes=CHECK,POINTER-TBAA %s

// cwg158: yes

// CHECK-LABEL: define {{.*}} @_Z1f
const int *f(const int * const *p, int **q) {
  // CHECK: load ptr, ptr %p.addr
  // CHECK: load ptr, {{.*}}, !tbaa ![[INTPTR_TBAA:[^,]*]]
  const int *x = *p;
  // CHECK: store ptr null, {{.*}}, !tbaa ![[INTPTR_TBAA]]
  *q = 0;
  return x;
}

struct A {};

// CHECK-LABEL: define {{.*}} @_Z1g
const int *(A::*const *g(const int *(A::* const **p)[3], int *(A::***q)[3]))[3] {
  // CHECK: load ptr, ptr %p.addr
  // CHECK: load ptr, {{.*}}, !tbaa ![[MEMPTR_TBAA:[^,]*]]
  const int *(A::*const *x)[3] = *p;
  // CHECK: store ptr null, {{.*}}, !tbaa ![[MEMPTR_TBAA]]
  *q = 0;
  return x;
}

// CHECK-LABEL: define {{.*}} @_Z1h
const int * h(const int * (*p)[10],  int *(*q)[9]) {
  // CHECK:  load ptr, ptr %p.addr, align 8, !tbaa [[PTRARRAY_TBAA:!.+]]
  const int * x = *p[0];

  // CHECK: load ptr, ptr %q.addr, align 8, !tbaa [[PTRARRAY_TBAA]]
  *q[0] = 0;
  return x;
}

// POINTER-TBAA: [[PTRARRAY_TBAA]] = !{[[PTRARRAY_TY:!.+]], [[PTRARRAY_TY]], i64 0}
// POINTER-TBAA: [[PTRARRAY_TY]] = !{!"p2 int", [[ANYPTR:!.+]], i64 0}
// POINTER-TBAA: [[ANYPTR]] = !{!"any pointer"
