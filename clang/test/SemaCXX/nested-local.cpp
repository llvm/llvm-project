// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s

void fn() {
  struct Outer {
    struct Inner {
      void foo(auto x) {} // expected-error {{'auto' not allowed in function prototype}}
    };
  };
}
