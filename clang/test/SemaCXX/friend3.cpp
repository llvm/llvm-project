// RUN: %clang_cc1 -S -triple i686-pc-linux-gnu -std=c++11 %s -o - | FileCheck %s

namespace pr8852 {
void foo();
struct S {
  friend void foo() {}
};

void main() {
  foo();
}
// CHECK: _ZN6pr88523fooEv:
}

namespace pr9518 {
template<typename T>
struct provide {
  friend T f() { return T(); }
};

void g() {
  void f();
  provide<void> p;
  f();
}
// CHECK: _ZN6pr95181fEv:
}
