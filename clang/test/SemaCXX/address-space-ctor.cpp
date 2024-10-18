// RUN: %clang_cc1 %s -std=c++14 -triple=spir -verify -fsyntax-only
// RUN: %clang_cc1 %s -std=c++17 -triple=spir -verify -fsyntax-only

// expected-no-diagnostics

struct MyType {
  MyType(int i) : i(i) {}
  int i;
};

MyType __attribute__((address_space(10))) m1 = 123;
MyType __attribute__((address_space(10))) m2(123);
