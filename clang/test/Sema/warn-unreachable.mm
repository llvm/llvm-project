// RUN: %clang_cc1 -fsyntax-only -fobjc-exceptions -fcxx-exceptions -verify -Wunreachable-code %s

void f();

void g3() {
  try {
    @try {
      f();
      throw 4; // caught by @catch, not by outer c++ catch.
      f(); // expected-warning {{will never be executed}}
    } @catch (...) {
    }
    f(); // not-unreachable
  } catch (...) {
  }
}
