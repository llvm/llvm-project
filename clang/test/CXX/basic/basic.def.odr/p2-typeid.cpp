// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

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

#if __cplusplus >= 202002L

namespace unevaluated {

struct S {
  void f();
};
struct T {
  virtual void f();
};

consteval S *null_s() { return nullptr; }
consteval S *make_s() { return new S; }
consteval T *null_t() { return nullptr; }
consteval T *make_t() { return new T; } // #alloc

void func() {
  (void)typeid(*null_s());
  (void)typeid(*make_s());
  (void)typeid(*null_t()); // expected-warning {{expression with side effects will be evaluated despite being used as an operand to 'typeid'}}
  (void)typeid(*make_t()); // expected-error {{call to consteval function 'unevaluated::make_t' is not a constant expression}} \
                              expected-note {{pointer to heap-allocated object is not a constant expression}} \
                              expected-note@#alloc {{heap allocation performed here}} \
                              expected-warning {{expression with side effects will be evaluated despite being used as an operand to 'typeid'}}
}

} // namespace unevaluated

#endif
