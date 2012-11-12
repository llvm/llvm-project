// RUN: %clang_cc1 -fsyntax-only -verify %s

class X {
public:
  typedef int I;
  class Y { };
  I a;
};

I b; // expected-error{{unknown type name 'I'}}
Y c; // expected-error{{unknown type name 'Y'}}
X::Y d;
X::I e;
