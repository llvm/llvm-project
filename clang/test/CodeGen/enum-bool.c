// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c23 -emit-llvm -o - %s | FileCheck %s

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
