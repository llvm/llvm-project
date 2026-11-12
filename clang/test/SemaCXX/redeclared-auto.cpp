// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

extern int a;
auto a = 0; // expected-note 2{{here}}
auto a = 0; // expected-error {{redefinition}}
int a = 0; // expected-error {{redefinition}}
extern auto a; // expected-error {{requires an initializer}}

extern int b; // expected-note {{here}}
auto b = 0.0; // expected-error {{different type}}

struct S {
  static int a;
  static int b; // expected-note {{here}}
};

auto S::a = 0; // expected-note 2{{here}}
auto S::a; // expected-error {{redefinition}} expected-error {{requires an initializer}}
int S::a = 0; // expected-error {{redefinition}}

auto S::b = 0.0; // expected-error {{different type}}

void f() {
  extern int a;
  extern auto a; // expected-error {{requires an initializer}}
}
