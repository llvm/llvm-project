// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c23 -ffixed-point -emit-llvm -o - %s | FileCheck %s

// C23 6.7.3.3p16: conversion to an enumerated type with a fixed underlying
// type has the same semantics as conversion to the underlying type. For an
// underlying type of bool, any nonzero value converts to true, so the value
// has to be compared against zero rather than truncated.

enum E : bool { Zero, One };

enum E a(int x) { return (enum E)x; }
// CHECK-LABEL: define{{.*}} zeroext i1 @a(
// CHECK-NOT: trunc
// CHECK: %tobool = icmp ne i32 %{{.*}}, 0
// CHECK: ret i1 %tobool

enum E b(int x) { return x; }
// CHECK-LABEL: define{{.*}} zeroext i1 @b(
// CHECK-NOT: trunc
// CHECK: %tobool = icmp ne i32 %{{.*}}, 0
// CHECK: ret i1 %tobool

enum E c(double x) { return (enum E)x; }
// CHECK-LABEL: define{{.*}} zeroext i1 @c(
// CHECK-NOT: fptoui
// CHECK: %tobool = fcmp une double %{{.*}}, 0.000000e+00
// CHECK: ret i1 %tobool

enum E d(double x) { enum E e = x; return e; }
// CHECK-LABEL: define{{.*}} zeroext i1 @d(
// CHECK-NOT: fptoui
// CHECK: %tobool = fcmp une double %{{.*}}, 0.000000e+00

enum E f(long long x) { enum E e; e = x; return e; }
// CHECK-LABEL: define{{.*}} zeroext i1 @f(
// CHECK-NOT: trunc
// CHECK: %tobool = icmp ne i64 %{{.*}}, 0

enum E g(_Complex double x) { return (enum E)x; }
// CHECK-LABEL: define{{.*}} zeroext i1 @g(
// CHECK-NOT: fptoui
// CHECK: fcmp une double
// CHECK: fcmp une double
// CHECK: %[[OR:.*]] = or i1
// CHECK: ret i1 %[[OR]]

// Compound assignment converts the result back to the enumeration type.
enum E h(enum E e, int x) { e += x; return e; }
// CHECK-LABEL: define{{.*}} zeroext i1 @h(
// CHECK: %[[ADD:.*]] = add nsw i32
// CHECK-NOT: trunc
// CHECK: icmp ne i32 %[[ADD]], 0

enum E i(enum E e, double x) { e += x; return e; }
// CHECK-LABEL: define{{.*}} zeroext i1 @i(
// CHECK: %[[ADD:.*]] = fadd double
// CHECK-NOT: fptoui
// CHECK: fcmp une double %[[ADD]], 0.000000e+00

enum E j(enum E e, _Complex double x) { e += x; return e; }
// CHECK-LABEL: define{{.*}} zeroext i1 @j(
// CHECK-NOT: fptoui
// CHECK: fcmp une double
// CHECK: fcmp une double
// CHECK: or i1

enum E k(enum E e, _Accum x) { e += x; return e; }
// CHECK-LABEL: define{{.*}} zeroext i1 @k(
// CHECK: add i47
// CHECK: %[[SUM:.*]] = trunc i47 %{{.*}} to i32
// CHECK: icmp ne i32 %[[SUM]], 0

// Like bool, increment always gives true; decrement flips the value.
enum E l(enum E e) { return ++e; }
// CHECK-LABEL: define{{.*}} zeroext i1 @l(
// CHECK-NOT: = add
// CHECK: store i8 1, ptr
// CHECK: ret i1 true

enum E m(enum E e) { e++; return e; }
// CHECK-LABEL: define{{.*}} zeroext i1 @m(
// CHECK-NOT: = add
// CHECK: store i8 1, ptr

enum E n(enum E e) { return --e; }
// CHECK-LABEL: define{{.*}} zeroext i1 @n(
// CHECK: %[[DEC:.*]] = add i1 %{{.*}}, true
// CHECK: ret i1 %[[DEC]]

void o(_Atomic(enum E) *p) { ++*p; }
// CHECK-LABEL: define{{.*}} void @o(
// CHECK-NOT: atomicrmw add
// CHECK: store atomic i8 1, ptr %{{.*}} seq_cst
