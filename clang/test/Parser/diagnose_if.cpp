// RUN: %clang_cc1 %s -fsyntax-only -fcxx-exceptions -verify

void t1() __attribute__((__diagnose_if__(baz))) try {} catch(...) {}
// expected-error@-1 {{use of undeclared identifier 'baz'}}

struct A {
  A();
};

A::A() __attribute__((__diagnose_if__(baz))) :;
// expected-error@-1 {{expected class member or base class name}}
// expected-error@-2 {{use of undeclared identifier 'baz'}}
