// RUN: %clang_cc1 -std=c++2a -verify %s

// When forming and checking satisfaction of atomic constraints, we will
// substitute still-dependent template arguments into an expression, and later
// substitute into the result. This creates some unique situations; check that
// they work.

namespace SubstIntoResolvedTypeTemplateArg {
  template<int, class> struct X {};

  template<class T> concept A = true;
  template<class T> concept B = sizeof(T) != 0;
  template<class T> concept C = B<X<1, T>>;

  int f(A auto); // expected-note {{candidate}}
  int f(C auto); // expected-note {{candidate}}
  int k1 = f(0); // expected-error {{ambiguous}}

  template<class T> concept D = A<T> && B<X<1, T>>;
  int f(D auto);
  int k2 = f(0); // ok

  // The atomic constraint formed from B<X<(int)'\1', T>> is identical to the
  // one formed from C, even though the template arguments are written as
  // different expressions; the "equivalent" rules are used rather than the
  // "identical" rules when matching template arguments in concept-ids.
  template<class T> concept E = A<T> && B<X<(int)'\1', T>>;
  int g(C auto);
  int g(E auto); // expected-note {{candidate}}
  int k3 = g(0);

  int g(D auto); // expected-note {{candidate}}
  int k4 = g(0); // expected-error {{ambiguous}}
}
