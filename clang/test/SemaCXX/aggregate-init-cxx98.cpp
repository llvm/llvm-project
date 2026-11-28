// RUN: %clang_cc1 -std=c++98 -verify %s

// expected-no-diagnostics

struct A {
  A() = default;
  int n;
};
A a = {0};
