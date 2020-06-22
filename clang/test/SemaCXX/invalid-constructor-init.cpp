// RUN: %clang_cc1 -frecovery-ast -verify %s

struct X {
  int Y;
  constexpr X() // expected-error {{constexpr constructor never produces}}
      : Y(foo()) {} // expected-error {{use of undeclared identifier 'foo'}}
};
// no crash on evaluating the constexpr ctor.
constexpr int Z = X().Y; // expected-error {{constexpr variable 'Z' must be initialized by a constant expression}}

struct X2 {
  int Y = foo();    // expected-error {{use of undeclared identifier 'foo'}} \
                 // expected-note {{subexpression not valid in a constant expression}}
  constexpr X2() {} // expected-error {{constexpr constructor never produces a constant expression}}
};

struct X3 {
  int Y;
  constexpr X3() // expected-error {{constexpr constructor never produces}}
      : Y(({foo();})) {} // expected-error {{use of undeclared identifier 'foo'}}
};

struct CycleDelegate {
  int Y;
  CycleDelegate(int)
      : Y(foo()) {} // expected-error {{use of undeclared identifier 'foo'}}
  // no bogus "delegation cycle" diagnostic
  CycleDelegate(float) : CycleDelegate(1) {}
};
