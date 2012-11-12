// RUN: %clang_cc1 -verify -std=c++11 %s

namespace N {
  typedef char C;
}

namespace M {
  typedef double D;
}

struct NonLiteral {
  NonLiteral() {}
  NonLiteral(int) {} // expected-note 2{{here}}
  operator int() const { return 0; }
};
struct Literal {
  constexpr Literal() {}
  operator int() const { return 0; }
};

struct S {
  virtual int ImplicitlyVirtual() const;
};
struct T {};

template<typename T> struct ImplicitVirtualFromDependentBase : T {
  constexpr int ImplicitlyVirtual() { return 0; }
};

constexpr int a = ImplicitVirtualFromDependentBase<S>().ImplicitlyVirtual(); // expected-error {{constant expression}} expected-note {{cannot evaluate virtual function call}}
constexpr int b = ImplicitVirtualFromDependentBase<T>().ImplicitlyVirtual(); // ok
constexpr int c = ImplicitVirtualFromDependentBase<S>().ImplicitVirtualFromDependentBase<S>::ImplicitlyVirtual();

template<typename R> struct ConstexprMember {
  constexpr R F() { return 0; }
};
constexpr int d = ConstexprMember<int>().F(); // ok
constexpr int e = ConstexprMember<NonLiteral>().F(); // expected-error {{constant expression}}

template<typename ...P> struct ConstexprCtor {
  constexpr ConstexprCtor(P...) {}
};
constexpr ConstexprCtor<> f1() { return {}; } // ok
constexpr ConstexprCtor<int> f2() { return 0; } // ok
constexpr ConstexprCtor<NonLiteral> f3() { return { 0 }; } // expected-error {{never produces a constant expression}} expected-note {{non-constexpr constructor 'NonLiteral}}
constexpr ConstexprCtor<int, NonLiteral> f4() { return { 0, 0 }; } // expected-error {{never produces a constant expression}} expected-note {{non-constexpr constructor 'NonLiteral}}

struct VirtBase : virtual S {}; // expected-note {{here}}

namespace TemplateVBase {
  template<typename T> struct T1 : virtual Literal { // expected-note {{here}}
    constexpr T1() {} // expected-error {{constexpr constructor not allowed in struct with virtual base class}}
  };

  template<typename T> struct T2 : virtual T {
    // FIXME: This is ill-formed (no diagnostic required).
    // We should diagnose it now rather than waiting until instantiation.
    constexpr T2() {}
  };
  constexpr T2<Literal> g2() { return {}; }

  template<typename T> class T3 : public T { // expected-note {{class with virtual base class is not a literal type}}
  public:
    constexpr T3() {}
  };
  constexpr T3<Literal> g3() { return {}; } // ok
  constexpr T3<VirtBase> g4() { return {}; } // expected-error {{not a literal type}}
}
