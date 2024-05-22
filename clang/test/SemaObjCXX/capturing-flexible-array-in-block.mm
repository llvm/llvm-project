// RUN: %clang_cc1 -fsyntax-only -fblocks -verify -std=c++11 %s

void f() {
  struct { int x; int y[]; } a; // expected-note 3 {{'a' declared here}}
  ^{return a.x;}(); // expected-error {{cannot refer to declaration of structure variable with flexible array member inside block}}
  [=] {return a.x;}(); // expected-error {{variable 'a' with flexible array member cannot be captured in a lambda expression}}
  [] {return a.x;}(); // expected-error {{variable 'a' cannot be implicitly captured in a lambda with no capture-default}} expected-note {{here}} expected-note 2 {{capture 'a' by}} expected-note 2 {{default capture by}}
}
