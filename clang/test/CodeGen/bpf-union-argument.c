// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -O2 -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

union t1 {};
union t2 {
  int a;
  long b;
};
union t3 {
  struct {
    int a;
    long b;
  };
  long c;
};
union t4 {
  struct {
    long a;
    long b;
    long c;
  };
  long d;
};

int foo1(union t1 arg1, union t2 arg2) {
// CHECK: define dso_local i32 @foo1(i64 %arg2.coerce)
  return arg2.a;
}

int foo2(union t3 arg1, union t4 arg2) {
// CHECK: define dso_local i32 @foo2([2 x i64] %arg1.coerce, ptr noundef byval(%union.t4) align 8 %arg2)
  return arg1.a + arg2.a;

}

int foo3(void) {
  union t1 tmp1 = {};
  union t2 tmp2 = {};
  union t3 tmp3 = {};
  union t4 tmp4 = {};
  return foo1(tmp1, tmp2) + foo2(tmp3, tmp4);
// CHECK: call i32 @foo1(i64 %{{[a-zA-Z0-9]+}})
// CHECK: call i32 @foo2([2 x i64] %{{[a-zA-Z0-9]+}}, ptr noundef byval(%union.t4) align 8 %tmp4)
}
