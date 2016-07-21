// RUN: %clang_cc1 %s -verify -fsyntax-only -triple=i686-linux-gnu -std=c++11

// We crashed when we couldn't properly convert the first arg of __atomic_* to
// an lvalue.
void PR28623() {
  void helper(int); // expected-note{{target}}
  void helper(char); // expected-note{{target}}
  __atomic_store_n(helper, 0, 0); // expected-error{{reference to overloaded function could not be resolved}}
}
