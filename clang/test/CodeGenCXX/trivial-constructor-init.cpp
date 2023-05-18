// RUN: %clang_cc1 -emit-llvm %s -o - -std=c++11 -triple %itanium_abi_triple | FileCheck %s

extern "C" int printf(...);

struct S {
  S() { printf("S::S\n"); }
};

struct A {
  double x;
  A() : x(), y(), s() { printf("x = %f y = %x \n", x, y); }
  int *y;
  S s;
};

A a;

struct B {
  B() = default;
  B(const B&);
};

// CHECK-NOT: _ZL1b
static B b;

struct C {
  ~C();
};

// CHECK: _ZL1c
static C c[4];

int main() {
}

namespace PR22793 {
template <typename>
struct foo {
protected:
// CHECK-NOT: _ZN7PR227933fooIiED2Ev
  ~foo() = default;
  friend void func();
};

void func() { foo<int> f; }

template struct foo<int>;
}
