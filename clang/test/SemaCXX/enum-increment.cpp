// RUN: %clang_cc1 -fsyntax-only %s -verify

enum A { A1, A2, A3 };
void test() {
  A a;
  a++;  // expected-error{{cannot increment expression of enum type 'A'}}
  a--;  // expected-error{{cannot decrement expression of enum type 'A'}}
  ++a;  // expected-error{{cannot increment expression of enum type 'A'}}
  --a;  // expected-error{{cannot decrement expression of enum type 'A'}}
}

enum B {B1, B2};
inline B &operator++ (B &b) { b = B((int)b+1); return b; }
inline B operator++ (B &b, int) { B ret = b; ++b; return b; }

void foo(enum B b) { ++b; b++; }
