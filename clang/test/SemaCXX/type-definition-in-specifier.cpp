// RUN: %clang_cc1 -fsyntax-only -verify %s

struct S0;
struct S1;
struct S2;
struct S3;
struct S4;
struct S5;
struct S6;

struct S0 { int x; };

void f0() {
  typedef struct S1 { int x; } S1_typedef;

  (void)((struct S2 { int x; }*)0); // expected-error{{can not be defined}}

  struct S3 { int x; } s3;

  (void)static_cast<struct S4 { int x; } *>(0); // expected-error{{can not be defined}}
}

struct S5 { int x; } f1() { return S5(); } // expected-error{{result type}}

void f2(struct S6 { int x; } p); // expected-error{{parameter type}}
