// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

int n;
struct S {
  int &a; // expected-note 2{{here}}
  int &b = n;

  union {
    const int k = 42;
  };

  S() {} // expected-error {{constructor for 'S' must explicitly initialize the reference member 'a'}}
  S(int) : a(n) {} // ok
  S(char) : b(n) {} // expected-error {{constructor for 'S' must explicitly initialize the reference member 'a'}}
  S(double) : a(n), b(n) {} // ok
} s(0);

union U {
  int a = 0; // desired-note 5 {{previous initialization is here}}
  char b = 'x';

  // FIXME: these should all be rejected
  U() {} // desired-error {{initializing multiple members of union}}
  U(int) : a(1) {} // desired-error {{initializing multiple members of union}}
  U(char) : b('y') {} // desired-error {{initializing multiple members of union}}
  // this expected note should be removed & the note should appear on the 
  // declaration of 'a' when this set of cases is handled correctly.
  U(double) : a(1), // expected-note{{previous initialization is here}} desired-error {{initializing multiple members of union}}
              b('y') {} // expected-error{{initializing multiple members of union}}
};

// PR10954: variant members do not acquire an implicit initializer.
namespace VariantMembers {
  struct NoDefaultCtor {
    NoDefaultCtor(int);
  };
  union V {
    NoDefaultCtor ndc;
    int n;

    V() {}
    V(int n) : n(n) {}
    V(int n, bool) : ndc(n) {}
  };
  struct K {
    union {
      NoDefaultCtor ndc;
      int n;
    };
    K() {}
    K(int n) : n(n) {}
    K(int n, bool) : ndc(n) {}
  };
  struct Nested {
    Nested() {}
    union {
      struct {
        NoDefaultCtor ndc;
      };
    };
  };
}
