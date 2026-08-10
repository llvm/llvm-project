// RUN: %clang_cc1 -std=c++2a -verify -triple=x86_64-linux-gnu %s
// expected-no-diagnostics

// Fixes assertion triggered by https://github.com/llvm/llvm-project/issues/65982

struct A { int y; };
struct B : virtual public A {};
struct X : public B {};

void member_with_virtual_inheritance() {
  X x;
  x.B::y = 1;
}
