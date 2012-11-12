// RUN: %clang_cc1 -fsyntax-only -verify %s

// C++ [basic.def.odr]p2:
//   An expression is potentially evaluated unless it [...] is the
//   operand of the typeid operator and the expression does not
//   designate an lvalue of polymorphic class type.

// FIXME: This should really include <typeinfo>, but we don't have that yet.
namespace std {
  class type_info;
}

struct Poly {
  virtual ~Poly();
};

struct NonPoly { };

template<typename T, typename Result = T> 
struct X {
  Result f(T t) { return t + t; } // expected-error{{invalid operands}}

  void g(T t) {
    (void)typeid(f(t)); // expected-note{{here}}
  }
};

void test(X<Poly> xp, X<Poly, Poly&> xpr, X<NonPoly> xnp, X<NonPoly, NonPoly&> xnpr) {
  // These are okay (although GCC and EDG get them wrong).
  xp.g(Poly());
  xnp.g(NonPoly());
  xnpr.g(NonPoly());

  // Triggers an error (as it should);
  xpr.g(Poly()); // expected-note{{instantiation of member function}}
}
