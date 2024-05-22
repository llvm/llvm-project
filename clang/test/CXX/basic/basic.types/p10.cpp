// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1y %s -DCXX1Y

struct NonLiteral { NonLiteral(); };

// A type is a literal type if it is:

// [C++1y] - void
constexpr void f() {}
#ifndef CXX1Y
// expected-error@-2 {{'void' is not a literal type}}
#endif

// - a scalar type
constexpr int f1(double) { return 0; }

// - a reference type
struct S { S(); };
constexpr int f2(S &) { return 0; }

struct BeingDefined;
extern BeingDefined beingdefined;
struct BeingDefined {
  static constexpr BeingDefined& t = beingdefined;
};

// - a class type that has all of the following properties:

// (implied) - it is complete

struct Incomplete; // expected-note 2{{forward declaration of 'Incomplete'}}
template<class T> struct ClassTemp {};

constexpr Incomplete incomplete = {}; // expected-error {{constexpr variable cannot have non-literal type 'const Incomplete'}} expected-note {{incomplete type 'const Incomplete' is not a literal type}}
constexpr Incomplete incomplete2[] = {}; // expected-error {{constexpr variable cannot have non-literal type 'const Incomplete[]'}} expected-note {{incomplete type 'const Incomplete[]' is not a literal type}}
constexpr ClassTemp<int> classtemplate = {};
constexpr ClassTemp<int> classtemplate2[] = {};

//  - it has a trivial destructor
struct UserProvDtor {
  ~UserProvDtor(); // expected-note {{has a user-provided destructor}}
};
constexpr int f(UserProvDtor) { return 0; } // expected-error {{'UserProvDtor' is not a literal type}}
struct NonTrivDtor {
  constexpr NonTrivDtor();
  virtual ~NonTrivDtor() = default; // expected-note {{has a non-trivial destructor}} expected-note {{because it is virtual}}
};
constexpr int f(NonTrivDtor) { return 0; } // expected-error {{'NonTrivDtor' is not a literal type}}
struct NonTrivDtorBase {
  ~NonTrivDtorBase();
};
template<typename T>
struct DerivedFromNonTrivDtor : T { // expected-note {{'DerivedFromNonTrivDtor<NonTrivDtorBase>' is not literal because it has base class 'NonTrivDtorBase' of non-literal type}}
  constexpr DerivedFromNonTrivDtor();
};
constexpr int f(DerivedFromNonTrivDtor<NonTrivDtorBase>) { return 0; } // expected-error {{constexpr function's 1st parameter type 'DerivedFromNonTrivDtor<NonTrivDtorBase>' is not a literal type}}
struct TrivDtor {
  constexpr TrivDtor();
};
constexpr int f(TrivDtor) { return 0; }
struct TrivDefaultedDtor {
  constexpr TrivDefaultedDtor();
  ~TrivDefaultedDtor() = default;
};
constexpr int f(TrivDefaultedDtor) { return 0; }

//  - it is an aggregate type or has at least one constexpr constructor or
//    constexpr constructor template that is not a copy or move constructor
struct Agg {
  int a;
  char *b;
};
constexpr int f3(Agg a) { return a.a; }
struct CtorTemplate {
  template<typename T> constexpr CtorTemplate(T);
};
struct CopyCtorOnly { // expected-note {{'CopyCtorOnly' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}
  constexpr CopyCtorOnly(CopyCtorOnly&);
};
constexpr int f(CopyCtorOnly) { return 0; } // expected-error {{'CopyCtorOnly' is not a literal type}}
struct MoveCtorOnly { // expected-note {{no constexpr constructors other than copy or move constructors}}
  constexpr MoveCtorOnly(MoveCtorOnly&&);
};
constexpr int f(MoveCtorOnly) { return 0; } // expected-error {{'MoveCtorOnly' is not a literal type}}
template<typename T>
struct CtorArg {
  constexpr CtorArg(T);
};
constexpr int f(CtorArg<int>) { return 0; } // ok
constexpr int f(CtorArg<NonLiteral>) { return 0; } // ok, ctor is still constexpr
// We have a special-case diagnostic for classes with virtual base classes.
struct VBase {};
struct HasVBase : virtual VBase {}; // expected-note 2{{virtual base class declared here}}
struct Derived : HasVBase {
  constexpr Derived() {} // expected-error {{constexpr constructor not allowed in struct with virtual base class}}
};
template<typename T> struct DerivedFromVBase : T { // expected-note {{struct with virtual base class is not a literal type}}
  constexpr DerivedFromVBase();
};
constexpr int f(DerivedFromVBase<HasVBase>) {} // expected-error {{constexpr function's 1st parameter type 'DerivedFromVBase<HasVBase>' is not a literal type}}
template<typename T> constexpr DerivedFromVBase<T>::DerivedFromVBase() : T() {}
constexpr int nVBase = (DerivedFromVBase<HasVBase>(), 0); // expected-error {{constant expression}} expected-note {{cannot construct object of type 'DerivedFromVBase<HasVBase>' with virtual base class in a constant expression}}

//  - it has all non-static data members and base classes of literal types
struct NonLitMember {
  S s; // expected-note {{has data member 's' of non-literal type 'S'}}
};
constexpr int f(NonLitMember) {} // expected-error {{1st parameter type 'NonLitMember' is not a literal type}}
struct NonLitBase :
  S { // expected-note {{base class 'S' of non-literal type}}
  constexpr NonLitBase();
};
constexpr int f(NonLitBase) { return 0; } // expected-error {{'NonLitBase' is not a literal type}}
struct LitMemBase : Agg {
  Agg agg;
};
template<typename T>
struct MemberType {
  T t; // expected-note {{'MemberType<NonLiteral>' is not literal because it has data member 't' of non-literal type 'NonLiteral'}}
  constexpr MemberType();
};
constexpr int f(MemberType<int>) { return 0; }
constexpr int f(MemberType<NonLiteral>) { return 0; } // expected-error {{not a literal type}}

// - an array of literal type [C++1y] other than an array of runtime bound
struct ArrGood {
  Agg agg[24];
  double d[12];
  TrivDtor td[3];
  TrivDefaultedDtor tdd[3];
};
constexpr int f(ArrGood) { return 0; }

struct ArrBad {
  S s[3]; // expected-note {{data member 's' of non-literal type 'S[3]'}}
};
constexpr int f(ArrBad) { return 0; } // expected-error {{1st parameter type 'ArrBad' is not a literal type}}

constexpr int arb(int n) { // expected-note {{declared here}}
  int a[n]; // expected-error {{variable of non-literal type 'int[n]' cannot be defined in a constexpr function}} \
               expected-warning {{variable length arrays in C++ are a Clang extension}} \
               expected-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
}
constexpr long Overflow[(1 << 30) << 2]{}; // expected-warning {{requires 34 bits to represent}} \
                                              expected-warning {{variable length array folded to constant array as an extension}} \
                                              expected-warning {{variable length arrays in C++ are a Clang extension}} \
                                              expected-note {{signed left shift discards bits}}

namespace inherited_ctor {
  struct A { constexpr A(int); };
  struct B : A {
    B();
    using A::A;
  };
  constexpr int f(B) { return 0; } // ok

  struct C { constexpr C(int); };
  struct D : C { // expected-note {{because}}
    D(int);
    using C::C;
  };
  constexpr int f(D) { return 0; } // expected-error {{not a literal type}}

  // This one is a bit odd: F inherits E's default constructor, which is
  // constexpr. Because F has a constructor of its own, it doesn't declare a
  // default constructor hiding E's one.
  struct E {};
  struct F : E {
    F(int);
    using E::E;
  };
  constexpr int f(F) { return 0; }

  // FIXME: Is this really the right behavior? We presumably should be checking
  // whether the inherited constructor would be a copy or move constructor for
  // the derived class, not for the base class.
  struct G { constexpr G(const G&); };
  struct H : G { // expected-note {{because}}
    using G::G;
  };
  constexpr int f(H) { return 0; } // expected-error {{not a literal type}}

  struct J;
  struct I { constexpr I(const J&); };
  struct J : I {
    using I::I;
  };
  constexpr int f(J) { return 0; }
}
