// RUN: %clang_cc1 -verify %s

struct S {
  typedef struct A {} A; // expected-note {{previous definition is here}}
  typedef struct B B;
  typedef A A; // expected-error {{redefinition of 'A'}}

  struct C { };
  typedef struct C OtherC;
  typedef OtherC C;

  typedef struct D { } D2;
  typedef D2 D;
};

