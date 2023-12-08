// RUN: %clang_cc1 -frecovery-ast -verify %s

struct X {
  int Y;
  constexpr X()
      : Y(foo()) {} // expected-error {{use of undeclared identifier 'foo'}}
};
// no crash on evaluating the constexpr ctor.
constexpr int Z = X().Y; // expected-error {{constexpr variable 'Z' must be initialized by a constant expression}}

struct X2 {
  int Y = foo();    // expected-error {{use of undeclared identifier 'foo'}}
  constexpr X2() {}
};

struct X3 {
  int Y;
  constexpr X3()
      : Y(({foo();})) {} // expected-error {{use of undeclared identifier 'foo'}}
};

struct CycleDelegate {
  int Y;
  CycleDelegate(int)
      : Y(foo()) {} // expected-error {{use of undeclared identifier 'foo'}}
  // no bogus "delegation cycle" diagnostic
  CycleDelegate(float) : CycleDelegate(1) {}
};

struct X4 {
  int* p = new int(invalid()); // expected-error {{use of undeclared identifier}}
};
// no crash on evaluating the CXXDefaultInitExpr.
constexpr int* s = X4().p; // expected-error {{must be initialized by}}
