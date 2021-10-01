// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff  -fxl-pragma-pack -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff  -fxl-pragma-pack -verify -fsyntax-only %s

#pragma align(packed)
struct A {  // expected-warning {{#pragma align(packed) may not be compatible with objects generated with AIX XL C/C++}}
  short s1;
  int   : 0;
  short s2;
};

struct B {  // expected-warning {{#pragma align(packed) may not be compatible with objects generated with AIX XL C/C++}}
  short a : 8;
  short b : 8;
  int c;
};

struct C {
  int x, y, z;
};

struct D {
  double d;
  struct A a;
};
#pragma align(reset)

struct E {
  int a : 28;
  int   : 0;
  int b : 16;
};
