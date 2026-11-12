// RUN: %clang_cc1 -fsyntax-only -verify %s

template <typename A, typedef B> struct Foo {
  // expected-error@-1 {{expected template parameter}} expected-note@-1 {{did you mean to use 'typename'?}}

  // Check that we are speculatively (with fixit applied) trying to parse the rest.

  // Should not produce error about type since parsing speculatively with fixit applied.
  B member;

  a // expected-error {{unknown type name 'a'}} // expected-error@+1 {{expected member name or ';' after declaration specifiers}}
};


// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// CHECK: fix-it:{{.*}}:{3:23-3:30}:"typename"
