// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// From PR23057 comment #18 (https://llvm.org/bugs/show_bug.cgi?id=23057#c18).

namespace N {
  int X[10]; // expected-note{{declared here}}}}
}

void f1() {
  for (auto operator new : X); // expected-error{{'operator new' cannot be the name of a variable or data member}}
                               // expected-error@-1{{use of undeclared identifier 'X'; did you mean 'N::X'?}}
}

void f2() {
  for (a operator== :) // expected-error{{'operator==' cannot be the name of a variable or data member}}
                       // expected-error@-1{{expected expression}}
                       // expected-error@-2{{unknown type name 'a'}}
} // expected-error{{expected statement}}
