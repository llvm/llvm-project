// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused-value %s

// Make sure diagnostics that we don't print based on runtime control
// flow are delayed correctly in cases where we can't immediately tell whether
// the context is unevaluated.

namespace std {
  class type_info;
}

int& NP(int);
void test1() { (void)typeid(NP(1 << 32)); }

class Poly { virtual ~Poly(); };
Poly& P(int);
void test2() { (void)typeid(P(1 << 32)); } // expected-warning {{shift count >= width of type}}

void test3() { 1 ? (void)0 : (void)typeid(P(1 << 32)); }
