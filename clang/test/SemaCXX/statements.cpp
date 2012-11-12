// RUN: %clang_cc1 %s -fsyntax-only -pedantic -verify

void foo() { 
  return foo();
}

// PR6451 - C++ Jump checking
struct X {
  X();
};

void test2() {
  goto later;  // expected-error {{goto into protected scope}}
  X x;         // expected-note {{jump bypasses variable initialization}} 
later:
  ;
}

namespace PR6536 {
  struct A {};
  void a() { goto out; A x; out: return; }
}
