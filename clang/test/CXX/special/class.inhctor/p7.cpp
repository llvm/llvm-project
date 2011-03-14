// RUN: true
// Disabled for 2.9
//%clang_cc1 -std=c++0x -fsyntax-only -verify %s

// Straight from the standard
struct B1 {
  B1(int); // expected-note {{previous constructor}}
};
struct B2 {
  B2(int); // expected-note {{conflicting constructor}}
};
struct D1 : B1, B2 {
  using B1::B1; // expected-note {{inherited here}}
  using B2::B2; // expected-error {{already inherited constructor with the same signature}}
};
struct D2 : B1, B2 {
  using B1::B1;
  using B2::B2;
  D2(int);
};
