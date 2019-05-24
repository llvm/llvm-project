// RUN: %clang_cc1 -fsyntax-only -verify -x objective-c++ -std=c++11 %s

void foo() {  // expected-note {{to match this '{'}}
  int bar;
  auto baz = [
      bar(  // expected-note {{to match this '('}} expected-note {{to match this '('}}
        foo_undeclared() // expected-error{{use of undeclared identifier 'foo_undeclared'}}
      /* ) */
    ] () { };   // expected-error{{expected ')'}}
}               // expected-error{{expected ')'}} expected-error {{expected ',' or ']'}} expected-error{{expected ';' at end of declaration}} expected-error{{expected '}'}}
