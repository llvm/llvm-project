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

int foo1(struct t1 arg1, struct t2 arg2) {
// CHECK: define dso_local i32 @foo1(i32 %arg2.coerce)
  return arg2.a;
}

int foo2(struct t3 arg1, struct t4 arg2) {
// CHECK: define dso_local i32 @foo2([2 x i64] %arg1.coerce, ptr noundef byval(%struct.t4) align 8 %arg2)
  return arg1.a + arg2.a;
}

int foo3(void) {
  struct t1 tmp1 = {};
  struct t2 tmp2 = {};
  struct t3 tmp3 = {};
  struct t4 tmp4 = {};
  return foo1(tmp1, tmp2) + foo2(tmp3, tmp4);
// CHECK: call i32 @foo1(i32 %{{[a-zA-Z0-9]+}})
// CHECK: call i32 @foo2([2 x i64] %{{[a-zA-Z0-9]+}}, ptr noundef byval(%struct.t4) align 8 %tmp4)
}
