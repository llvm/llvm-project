// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -O2 -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s
//
// Aggregates up to 16 bytes are returned directly in registers: coerced to an
// integer when they fit in one register (<= 8 bytes), or to [2 x i64] when
// they need two (9..16 bytes).

struct foo1 {int a;};                             // 4 bytes  -> one register
struct foo2 {int a; long b;};                     // 16 bytes -> two registers
struct foo3 {int a; int b; long c;};              // 16 bytes -> two registers
struct foo4 {int a; int b:20; int c:20; int d:24;}; // 16 bytes -> two registers

#define __noinline __attribute__((noinline))

__noinline struct foo1 bar1(int a) {
// CHECK-LABEL: define dso_local i32 @bar1(
// CHECK: ret i32
  struct foo1 v = {a};
  return v;
}

__noinline struct foo2 bar2(int a, int b) {
// CHECK-LABEL: define dso_local [2 x i64] @bar2(
// CHECK: ret [2 x i64]
  struct foo2 v = {a, b};
  return v;
}

__noinline struct foo3 bar3(int a, int b, int c) {
// CHECK-LABEL: define dso_local [2 x i64] @bar3(
// CHECK: ret [2 x i64]
  struct foo3 v = {a, b, c};
  return v;
}

__noinline struct foo4 bar4(int a, int b, int c, int d) {
// CHECK-LABEL: define dso_local [2 x i64] @bar4(
// CHECK: ret [2 x i64]
  struct foo4 v = {a, b, c, d};
  return v;
}

int check1(int a) {
// CHECK-LABEL: define dso_local i32 @check1(
// CHECK: %[[C1:.*]] = call i32 @bar1(
// CHECK: store i32 %[[C1]]
  struct foo1 v1 = bar1(a);
  return v1.a;
}

int check2(int a, int b) {
// CHECK-LABEL: define dso_local i32 @check2(
// CHECK: %[[C2:.*]] = call [2 x i64] @bar2(
// CHECK: store [2 x i64] %[[C2]]
  struct foo2 v1 = bar2(a, b);
  return v1.a + v1.b;
}

int check3(int a, int b, int c) {
// CHECK-LABEL: define dso_local i32 @check3(
// CHECK: %[[C3:.*]] = call [2 x i64] @bar3(
// CHECK: store [2 x i64] %[[C3]]
  struct foo3 v1 = bar3(a, b, c);
  return v1.a + v1.b + v1.c;
}

int check4(int a, int b, int c, int d) {
// CHECK-LABEL: define dso_local i32 @check4(
// CHECK: %[[C4:.*]] = call [2 x i64] @bar4(
// CHECK: store [2 x i64] %[[C4]]
  struct foo4 v1 = bar4(a, b, c, d);
  return v1.a + v1.b + v1.c + v1.d;
}
