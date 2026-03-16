// RUN: %clang_cc1 -verify %s

int bar; // #bar

int foo()
{
  // FIXME: Bad error recovery.
  (void)(baz<> + baz<>);
  // expected-error@-1 2{{use of undeclared identifier 'baz'; did you mean 'bar'?}}
  // expected-note@#bar 2{{'bar' declared here}}
  // expected-error@-3 2{{'bar' is expected to be a non-type template, but instantiated to a class template}}
  // expected-note@#bar 2{{class template declared here}}
}
