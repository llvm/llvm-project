// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -DMS -fms-compatibility -Wmicrosoft-inaccessible-base -fsyntax-only -verify %s

struct A {
  int a;
};

struct X1 : virtual A
{};

struct Y1 : X1, virtual A
{};

struct Y2 : X1, A // expected-warning{{direct base 'A' is inaccessible due to ambiguity:\n    struct Y2 -> X1 -> A\n    struct Y2 -> A}}
{};

struct X2 : A
{};

struct Z1 : X2, virtual A // expected-warning{{direct base 'A' is inaccessible due to ambiguity:\n    struct Z1 -> X2 -> A\n    struct Z1 -> A}}
{};

struct Z2 : X2, A // expected-warning{{direct base 'A' is inaccessible due to ambiguity:\n    struct Z2 -> X2 -> A\n    struct Z2 -> A}}
{};

A *y2_to_a(Y2 *p) {
#ifdef MS
  // expected-warning@+4 {{accessing inaccessible direct base 'A' of 'Y2' is a Microsoft extension}}
#else
  // expected-error@+2 {{ambiguous conversion}}
#endif
  return p;
}

A *z1_to_a(Z1 *p) {
#ifdef MS
  // expected-warning@+4 {{accessing inaccessible direct base 'A' of 'Z1' is a Microsoft extension}}
#else
  // expected-error@+2 {{ambiguous conversion}}
#endif
  return p;
}

A *z1_to_a(Z2 *p) {
#ifdef MS
  // expected-warning@+4 {{accessing inaccessible direct base 'A' of 'Z2' is a Microsoft extension}}
#else
  // expected-error@+2 {{ambiguous conversion}}
#endif
  return p;
}
