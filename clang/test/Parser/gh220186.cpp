// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

void foo1() { (auto()->bar::); }
// expected-error@-1 {{use of undeclared identifier 'bar'}}
// expected-error@-2 {{initializer for functional-style cast to 'auto' is empty}}
// expected-error@-3 {{expected unqualified-id}}

void foo2() { (auto()->bar::~bar()); }
// expected-error@-1 {{use of undeclared identifier 'bar'}}
// expected-error@-2 {{initializer for functional-style cast to 'auto' is empty}}
