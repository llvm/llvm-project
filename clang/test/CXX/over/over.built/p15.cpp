// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

struct A { operator decltype(nullptr)(); };
struct B { operator const int *(); };
void f(A a, B b, volatile int *pi) {
  (void)(a == a);
  (void)(a != a);
  (void)(a < a); // expected-error {{invalid operands}}
  (void)(a > a); // expected-error {{invalid operands}}
  (void)(a <= a); // expected-error {{invalid operands}}
  (void)(a >= a); // expected-error {{invalid operands}}

  (void)(a == b);
  (void)(a != b);
  // FIXME: These cases were intended to be made ill-formed by N3624, but it
  // fails to actually achieve this goal.
  (void)(a < b);
  (void)(a > b);
  (void)(a <= b);
  (void)(a >= b);

  (void)(b == a);
  (void)(b != a);
  // FIXME: These cases were intended to be made ill-formed by N3624, but it
  // fails to actually achieve this goal.
  (void)(b < a);
  (void)(b > a);
  (void)(b <= a);
  (void)(b >= a);

  (void)(a == pi);
  (void)(a != pi);
  // FIXME: These cases were intended to be made ill-formed by N3624, but it
  // fails to actually achieve this goal.
  (void)(a < pi);
  (void)(a > pi);
  (void)(a <= pi);
  (void)(a >= pi);

  (void)(pi == a);
  (void)(pi != a);
  // FIXME: These cases were intended to be made ill-formed by N3624, but it
  // fails to actually achieve this goal.
  (void)(pi < a);
  (void)(pi > a);
  (void)(pi <= a);
  (void)(pi >= a);

  (void)(b == pi);
  (void)(b != pi);
  (void)(b < pi);
  (void)(b > pi);
  (void)(b <= pi);
  (void)(b >= pi);

  (void)(pi == b);
  (void)(pi != b);
  (void)(pi < b);
  (void)(pi > b);
  (void)(pi <= b);
  (void)(pi >= b);

  (void)(b == b);
  (void)(b != b);
  (void)(b < b);
  (void)(b > b);
  (void)(b <= b);
  (void)(b >= b);

  (void)(pi == pi);
  (void)(pi != pi);
  (void)(pi < pi);
  (void)(pi > pi);
  (void)(pi <= pi);
  (void)(pi >= pi);
}

// FIXME: This is wrong: the type T = 'const volatile int * const * const *'
// would work here, and there exists a builtin candidate for that type.
struct C { operator const int ***(); };
void g(C c, volatile int ***p) {
  (void)(c < p); // expected-error {{invalid operands}}
}
