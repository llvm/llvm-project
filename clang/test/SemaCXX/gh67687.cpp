// RUN: %clang_cc1 -verify -fsyntax-only -std=c++17 %s

struct S {
  int n;
  int d = (4, []() { return n; }());  // expected-error {{'this' cannot be implicitly captured in this context}} \
                                      // expected-note {{explicitly capture 'this'}}
};
