// RUN: %clang_cc1 -fsyntax-only -verify %s

class X {
public:
  typedef int I; // expected-note{{'X::I' declared here}}
  class Y { }; // expected-note{{'X::Y' declared here}}
  I a;
};

I b; // expected-error{{unknown type name 'I'; did you mean 'X::I'?}}
Y c; // expected-error{{unknown type name 'Y'; did you mean 'X::Y'?}}
X::Y d;
X::I e;
