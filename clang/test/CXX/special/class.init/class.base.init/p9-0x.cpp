// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++11 %s -O1 -emit-llvm -o - | FileCheck %s

struct S {
  int n = 10;
  int m = 2 * n;

  S() {}
  S(int a) : n(a) {}
  S(int a, int b) : n(a), m(b) {}

  struct T {
    T *that = this;
  };
};

template<typename T>
struct U {
  T *r = &q;
  T q = 42;
  U *p = this;
};

S a;
// CHECK: @a = {{.*}} { i32 10, i32 20 }

S b(5);
// CHECK: @b = {{.*}} { i32 5, i32 10 }

S c(3, 9);
// CHECK: @c = {{.*}} { i32 3, i32 9 }

S::T d;
// CHECK: @d = {{.*}} { {{.*}} @d }

U<S> e;
// CHECK: @e = {{.*}} { {{.*}} { i32 42, i32 84 }, {{.*}} @e }
