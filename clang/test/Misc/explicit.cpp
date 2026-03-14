// RUN: %clang_cc1 -std=c++20 %s -verify

int foo ()  {
  int b;
  explicit( && b );  // expected-error{{conversion from 'void *' to 'bool' is not allowed in a converted constant expression}}
                     // expected-error@-1{{'explicit' can only appear on non-static member functions}}
                     // expected-error@-2{{use of undeclared label 'b'}}
                     // expected-warning@-3{{declaration does not declare anything}}
}
