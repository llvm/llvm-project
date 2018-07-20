// RUN: %clang_cc1 %s -fsyntax-only -verify -Wduplicate-enum
// RUN: %clang_cc1 %s -x c++ -fsyntax-only -verify -Wduplicate-enum
enum A {
  A1 = 0,  // expected-note {{element 'A1' also has value 0}}
  A2 = -1,
  A3,  // expected-warning {{element 'A3' has been implicitly assigned 0 which another element has been assigned}}
  A4};

enum B {
  B1 = -1,  // expected-note {{element 'B1' also has value -1}}
  B2,       // expected-warning {{element 'B2' has been implicitly assigned 0 which another element has been assigned}}
  B3,
  B4 = -2,
  B5,  // expected-warning {{element 'B5' has been implicitly assigned -1 which another element has been assigned}}
  B6   // expected-note {{element 'B6' also has value 0}}
};

enum C { C1, C2 = -1, C3 }; // expected-warning{{element 'C1' has been implicitly assigned 0 which another element has been assigned}} \
  // expected-note {{element 'C3' also has value 0}}

enum D {
  D1,
  D2,
  D3,  // expected-warning{{element 'D3' has been implicitly assigned 2 which another element has been assigned}}
  D4 = D2,  // no warning
  D5 = 2  // expected-note {{element 'D5' also has value 2}}
};

enum E {
  E1,
  E2 = E1,
  E3 = E2
};

enum F {
  F1,
  F2,
  FCount,
  FMax = FCount - 1
};

enum G {
  G1,
  G2,
  GMax = G2,
  GCount = GMax + 1
};

enum {
  H1 = 0,
  H2 = -1,
  H3,
  H4};

enum {
  I1 = -1,
  I2,
  I3,
  I4 = -2,
  I5,
  I6
};

enum { J1, J2 = -1, J3 };

enum { 
  K1, 
  K2, 
  K3,
  K4 = K2,
  K5 = 2
};

enum {
  L1,
  L2 = L1,
  L3 = L2
};

enum {
  M1,
  M2,
  MCount,
  MMax = MCount - 1
};

enum {
  N1,
  N2,
  NMax = N2,
  NCount = NMax + 1
};

// PR15693
enum enum1 {
  VALUE // expected-note{{previous definition is here}}
};

enum enum2 {
  VALUE // expected-error{{redefinition of enumerator 'VALUE'}}
};
