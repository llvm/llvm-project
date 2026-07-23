// RUN: %clang_cc1 -std=c++26 -verify %s
// expected-no-diagnostics

void foo() {
  template for (auto x : {1, 3}) {
    template for (auto x : {1, 0}) { void bar(); }
  }
}
