// RUN: %clang_cc1 -fsyntax-only -verify %s
// XFAIL: *

class C {
public:
  C(int a, int b);
};

C::C(int a, // expected-note {{previous definition}}
     int b) // expected-note {{previous definition}}
try {
  int c;

} catch (int a) { // expected-error {{redefinition of 'a'}}
  int b; // expected-error {{redefinition of 'b'}}
  ++c; // expected-error {{use of undeclared identifier 'c'}}
}
