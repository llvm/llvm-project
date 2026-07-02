// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -O2 -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

struct t1 {};
struct t2 {
  int a;
};
struct t3 {
  int a;
  long b;
};
struct t4 {
  long a;
  long b;
  long c;
};
struct t5 {
  char a;
};
union u1 {
  int a;
  long b;
};

struct t1 foo1(void) {
// CHECK: define dso_local void @foo1()
  struct t1 tmp = {};
  return tmp;
}

struct t2 foo2(void) {
// CHECK: define dso_local i32 @foo2()
  struct t2 tmp = {};
  return tmp;
}

struct t3 foo3(void) {
// CHECK: define dso_local [2 x i64] @foo3()
  struct t3 tmp = {};
  return tmp;
}

struct t4 foo4(void) {
// CHECK: define dso_local void @foo4(ptr dead_on_unwind noalias writable sret(%struct.t4) align 8 %agg.result)
  struct t4 tmp = {};
  return tmp;
}

struct t5 foo5(void) {
// CHECK: define dso_local i8 @foo5()
  struct t5 tmp = {};
  return tmp;
}

union u1 foou(void) {
// CHECK: define dso_local i64 @foou()
  union u1 tmp = {};
  return tmp;
}

int bar(void) {
// CHECK-LABEL: define dso_local i32 @bar()
// CHECK: %[[C2:.*]] = call i32 @foo2()
// CHECK: store i32 %[[C2]]
// CHECK: %[[C3:.*]] = call [2 x i64] @foo3()
// CHECK: store [2 x i64] %[[C3]]
  struct t2 a = foo2();
  struct t3 b = foo3();
  return a.a + b.a;
}
