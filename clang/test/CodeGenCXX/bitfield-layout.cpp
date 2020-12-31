// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -O3 | FileCheck -check-prefix=CHECK -check-prefix=CHECK-LP64 %s
// RUN: %clang_cc1 %s -triple=i386-apple-darwin10 -emit-llvm -o - -O3 | FileCheck %s
// RUN: %clang_cc1 %s -triple=aarch64_be-none-eabi -emit-llvm -o - -O3 | FileCheck %s
// RUN: %clang_cc1 %s -triple=thumbv7_be-none-eabi -emit-llvm -o - -O3 | FileCheck %s

// CHECK-LP64: %union.Test1 = type { i32, [4 x i8] }
union Test1 {
  int a;
  int b: 39;
};
Test1 t1;

// CHECK-LP64: %union.Test2 = type { i8 }
union Test2 {
  int : 6;
} t2;

// CHECK-LP64: %union.Test3 = type { i16 }
union Test3 {
  int : 9;
} t3;

// CHECK: %union.Test4 = type { i8, i8 }
union Test4 {
  char val : 16;
};
Test4 t4;

#define CHECK(x) if (!(x)) return __LINE__

// CHECK: define{{.*}} i32 @_Z11test_assignv()
int test_assign() {
  struct {
    int a;

    unsigned long long b : 65;

    int c;
  } c;
  
  c.a = 0;
  c.b = (unsigned long long)-1;
  c.c = 0;

  CHECK(c.a == 0);
  CHECK(c.b == (unsigned long long)-1);
  CHECK(c.c == 0);

  Test1 u1;
  Test4 u2;

  u1.b = 1;
  u2.val = 42;

  CHECK(u1.b == 1);
  CHECK(u2.val == 42);

  // CHECK: ret i32 0
  return 0;
}

// CHECK: define{{.*}} i32 @_Z9test_initv()
int test_init() {
  struct S {
    int a;

    unsigned long long b : 65;

    int c;
  };
  S s1 = {1, 42, 0};

  CHECK(s1.a == 1);
  CHECK(s1.b == (unsigned long long)42);
  CHECK(s1.c == 0);

  Test1 u1 = {1};
  Test4 u2 = {42};

  CHECK(u1.a == 1);
  CHECK(u1.b == 1);
  CHECK(u2.val == 42);

  // CHECK: ret i32 0
  return 0;
}
