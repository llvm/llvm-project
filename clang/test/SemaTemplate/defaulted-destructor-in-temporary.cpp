// RUN: %clang_cc1 -std=c++11 -triple=x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s

// CHECK: define linkonce_odr {{.*}} @_ZN3StrD1Ev

class A {
public:
  ~A();
};
class Str {
  A d;

public:
  ~Str() = default;
};
class E {
  Str s;
  template <typename>
  void h() {
    s = {};
  }
  void f();
};
void E::f() {
  h<int>();
}
