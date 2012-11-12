// RUN: %clang_cc1 -fsyntax-only -verify -fobjc-arc %s

// Make sure the ARC auto-deduction of id* in unevaluated contexts
// works correctly in cases where we can't immediately tell whether the
// context is unevaluated.

namespace std {
  class type_info;
}

int& NP(void*);
void test1() { (void)typeid(NP((void*)(id*)0)); }

class Poly { virtual ~Poly(); };
Poly& P(void*);
void test2() { (void)typeid(P((void*)(id*)0)); } // expected-error {{pointer to non-const type 'id'}}
