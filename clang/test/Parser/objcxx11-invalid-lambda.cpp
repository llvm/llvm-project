// RUN: %clang_cc1 -fsyntax-only -verify -x objective-c++ -std=c++11 %s

void foo() {
  int bar;
  auto baz = [
      bar(  // expected-note 2{{to match this '('}}\
            // expected-warning {{captures are a C++14 extension}}
        foo_undeclared() // expected-error{{use of undeclared identifier 'foo_undeclared'}}
      /* ) */
    ] () { };   // expected-error 2{{expected ')'}}
}