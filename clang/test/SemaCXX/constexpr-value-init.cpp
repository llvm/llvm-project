// RUN: %clang_cc1 %s -Wno-uninitialized -std=c++11 -fsyntax-only -verify

struct A {
  constexpr A() : a(b + 1), b(a + 1) {} // expected-note 5{{outside its lifetime}}
  int a;
  int b;
};
struct B { // expected-note {{in call to 'A()'}}
  A a;
};

constexpr A a1; // expected-error {{constant expression}} expected-note {{in call to 'A()'}}
constexpr A a2 = A(); // expected-error {{constant expression}} expected-note {{in call to 'A()'}}
void f() {
  constexpr A a; // expected-error {{constant expression}} expected-note {{in call to 'A()'}}
}

constexpr B b1; // expected-error {{constant expression}} expected-note {{in call to 'B()'}}
constexpr B b2 = B(); // ok
static_assert(b2.a.a == 1, "");
static_assert(b2.a.b == 2, "");

struct C {
  int c;
};
struct D : C { int d; };
constexpr C c1; // expected-error {{without a user-provided default constructor}}
constexpr C c2 = C(); // ok
constexpr D d1; // expected-error {{without a user-provided default constructor}}
constexpr D d2 = D(); // ok with DR1452
static_assert(D().c == 0, "");
static_assert(D().d == 0, "");

struct V : virtual C {};
template<typename T> struct Z : T {
  constexpr Z() : V() {}
};
constexpr int n = Z<V>().c; // expected-error {{constant expression}} expected-note {{non-literal type 'Z<V>'}}

struct E { // expected-note {{in call to 'A()'}}
  A a[2];
};
constexpr E e1; // expected-error {{constant expression}} expected-note {{in call to 'E()'}}
constexpr E e2 = E();
static_assert(e2.a[0].a == 1, "");
static_assert(e2.a[0].b == 2, "");
static_assert(e2.a[1].a == 1, "");
static_assert(e2.a[1].b == 2, "");
