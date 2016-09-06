// RUN: %clang_cc1 -std=c++1z -verify %s -Wdeprecated

namespace {
  struct A {
    static constexpr int n = 0;
  };
  const int A::n; // expected-warning {{deprecated}}
}
