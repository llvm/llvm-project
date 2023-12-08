// RUN: %clang_cc1 -fsyntax-only -verify %s
struct S {   // expected-note{{to match this}}
  void f() { // expected-note{{to match this}}
 // expected-error@+1{{expected '}'}} expected-error@+1{{expected '}'}} expected-error@+1{{expected ';'}}
#pragma pack()
