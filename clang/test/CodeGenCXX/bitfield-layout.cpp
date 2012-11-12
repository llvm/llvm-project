// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -O3 | FileCheck -check-prefix LP64 %s
// RUN: %clang_cc1 %s -triple=i386-apple-darwin10 -emit-llvm -o - -O3 | FileCheck -check-prefix LP32 %s

// CHECK-LP64: %union.Test1 = type { i32, [4 x i8] }
union Test1 {
  int a;
  int b: 39;
} t1;

// CHECK-LP64: %union.Test2 = type { i8 }
union Test2 {
  int : 6;
} t2;

// CHECK-LP64: %union.Test3 = type { [2 x i8] }
union Test3 {
  int : 9;
} t3;


#define CHECK(x) if (!(x)) return __LINE__

int f() {
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

// CHECK-LP64: ret i32 0
// CHECK-LP32: ret i32 0
  return 0;
}
