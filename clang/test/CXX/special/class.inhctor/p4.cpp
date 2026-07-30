// RUN: %clang_cc1 -std=c++11 -verify %s
//
// Note: [class.inhctor] was removed by P0136R1. This tests the new behavior
// for the wording that used to be there.

template<int> struct X {};

// A[n inheriting] constructor [...] has the same access as the corresponding
// constructor [in the base class].
struct A {
public:
  A(X<0>) {}
protected:
  A(X<1>) {} // expected-note 2{{declared protected here}}
private:
  A(X<2>) {} // expected-note 2{{declared private here}}
  friend class FA;
};

struct B : A {
  using A::A;
  friend class FB;
};

B b0{X<0>{}};
B b1{X<1>{}}; // expected-error {{calling a protected constructor}}
B b2{X<2>{}}; // expected-error {{calling a private constructor}}

struct C : B {
  C(X<0> x) : B(x) {}
  C(X<1> x) : B(x) {}
};

struct FB {
  B b0{X<0>{}};
  B b1{X<1>{}};
};

struct FA : A {
  using A::A;
};
FA fa0{X<0>{}};
FA fa1{X<1>{}}; // expected-error {{calling a protected constructor}}
FA fa2{X<2>{}}; // expected-error {{calling a private constructor}}


// It is deleted if the corresponding constructor [...] is deleted.
struct G {
  G(int) = delete; // expected-note {{'G' has been explicitly marked deleted here}}
  template<typename T> G(T*) = delete; // expected-note {{'G<const char>' has been explicitly marked deleted here}}
};
struct H : G {
  using G::G;
};
H h1(5); // expected-error {{call to deleted constructor of 'H'}}
H h2("foo"); // expected-error {{call to deleted constructor of 'H'}}


// Core defect: It is also deleted if multiple base constructors generate the
// same signature.
namespace DRnnnn {
  struct A {
    constexpr A(int, float = 0) {}
    explicit A(int, int = 0) {} // expected-note {{candidate}}

    A(int, int, int = 0) = delete; // expected-note {{deleted}}
  };
  struct B : A {
    using A::A; // expected-note 2{{inherited here}}
  };

  constexpr B b0(0, 0.0f); // ok, constexpr
  B b1(0, 1); // expected-error {{call to constructor of 'B' is ambiguous}}
}
