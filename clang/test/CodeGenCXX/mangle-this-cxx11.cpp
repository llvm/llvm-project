// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s

struct B {
 template <class U> U f();
};

struct A {
 B b;
 // implicitly rewritten to (*this).b.f<U>()
 template <class U> auto f() -> decltype (b.f<U>());
 template <class U> auto g() -> decltype (this->b.f<U>());
};

int main() {
  A a;
  // CHECK: call i32 @_ZN1A1fIiEEDTcldtdtdefpT1b1fIT_EEEv
  a.f<int>();
  // CHECK: call i32 @_ZN1A1gIiEEDTcldtptfpT1b1fIT_EEEv
  a.g<int>();
}
