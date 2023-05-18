// RUN: %clang_cc1 -std=c++98 -verify %s

struct A {
  A() = default; // expected-warning {{C++11}}
  int n;
};
A a = {0};
