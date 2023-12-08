// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

struct A { operator decltype(nullptr)(); };
struct B { operator int A::*(); };
void f(A a, B b, int A::*pi) {
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
  (void)(b < pi); // expected-error {{invalid operands}}
  (void)(b > pi); // expected-error {{invalid operands}}
  (void)(b <= pi); // expected-error {{invalid operands}}
  (void)(b >= pi); // expected-error {{invalid operands}}

  (void)(pi == b);
  (void)(pi != b);
  (void)(pi < b); // expected-error {{invalid operands}}
  (void)(pi > b); // expected-error {{invalid operands}}
  (void)(pi <= b); // expected-error {{invalid operands}}
  (void)(pi >= b); // expected-error {{invalid operands}}

  (void)(b == b);
  (void)(b != b);
  (void)(b < b); // expected-error {{invalid operands}}
  (void)(b > b); // expected-error {{invalid operands}}
  (void)(b <= b); // expected-error {{invalid operands}}
  (void)(b >= b); // expected-error {{invalid operands}}

  (void)(pi == pi);
  (void)(pi != pi);
  (void)(pi < pi); // expected-error {{invalid operands}}
  (void)(pi > pi); // expected-error {{invalid operands}}
  (void)(pi <= pi); // expected-error {{invalid operands}}
  (void)(pi >= pi); // expected-error {{invalid operands}}
}

// FIXME: This is wrong: type T = 'const volatile int * const A::* const B::*'
// would work here, and there exists a builtin candidate for that type.
struct C { operator const int *A::*B::*(); };
void g(C c, volatile int *A::*B::*p) {
  (void)(c == p); // expected-error {{invalid operands}}
}
