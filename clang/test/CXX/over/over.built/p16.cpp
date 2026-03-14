// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

struct A { operator decltype(nullptr)(); }; // expected-note 16{{implicitly converted}}
struct B { operator const int *(); }; // expected-note 8{{implicitly converted}}
void f(A a, B b, volatile int *pi) {
  (void)(a == a);
  (void)(a != a);
  (void)(a < a); // expected-error {{invalid operands}}
  (void)(a > a); // expected-error {{invalid operands}}
  (void)(a <= a); // expected-error {{invalid operands}}
  (void)(a >= a); // expected-error {{invalid operands}}

  (void)(a == b);
  (void)(a != b);
  (void)(a < b); // expected-error {{invalid operands}}
  (void)(a > b); // expected-error {{invalid operands}}
  (void)(a <= b); // expected-error {{invalid operands}}
  (void)(a >= b); // expected-error {{invalid operands}}

  (void)(b == a);
  (void)(b != a);
  (void)(b < a); // expected-error {{invalid operands}}
  (void)(b > a); // expected-error {{invalid operands}}
  (void)(b <= a); // expected-error {{invalid operands}}
  (void)(b >= a); // expected-error {{invalid operands}}

  (void)(a == pi);
  (void)(a != pi);
  (void)(a < pi); // expected-error {{invalid operands}}
  (void)(a > pi); // expected-error {{invalid operands}}
  (void)(a <= pi); // expected-error {{invalid operands}}
  (void)(a >= pi); // expected-error {{invalid operands}}

  (void)(pi == a);
  (void)(pi != a);
  (void)(pi < a); // expected-error {{invalid operands}}
  (void)(pi > a); // expected-error {{invalid operands}}
  (void)(pi <= a); // expected-error {{invalid operands}}
  (void)(pi >= a); // expected-error {{invalid operands}}

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
