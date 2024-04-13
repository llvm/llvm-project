// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s
// RUN: %clang_cc1 -std=c++11 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s
// RUN: %clang_cc1 -std=c++14 %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s
// RUN: %clang_cc1 -std=c++1z %s -triple x86_64-linux-gnu -emit-llvm -o - -fexceptions -fcxx-exceptions -pedantic-errors | FileCheck %s

// cwg1748: 3.7

// FIXME: __SIZE_TYPE__ expands to 'long long' on some targets.
__extension__ typedef __SIZE_TYPE__ size_t;

void *operator new(size_t, void *);
void *operator new[](size_t, void *);

struct X { X(); };

// The reserved placement allocation functions get inlined
// even if we can't see their definitions. They do not
// perform a null check.

// CHECK-LABEL: define {{.*}} @_Z1fPv(
// CHECK-NOT: call
// CHECK-NOT: icmp{{.*}} null
// CHECK-NOT: br i1
// CHECK: call void @_ZN1XC1Ev(
// CHECK: }
X *f(void *p) { return new (p) X; }

// CHECK-LABEL: define {{.*}} @_Z1gPv(
// CHECK-NOT: call
// CHECK-NOT: icmp{{.*}} null
// CHECK-NOT: br i1
// CHECK: call void @_ZN1XC1Ev(
// CHECK: br i1
// CHECK: }
X *g(void *p) { return new (p) X[5]; }
