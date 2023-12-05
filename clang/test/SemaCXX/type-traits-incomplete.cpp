// RUN: %clang_cc1 -fsyntax-only -verify %s 

struct S; // expected-note 6 {{forward declaration of 'S'}}

void f() {
  __is_pod(S); // expected-error{{incomplete type 'S' used in type trait expression}}
  __is_pod(S[]); // expected-error{{incomplete type 'S' used in type trait expression}}

  __is_trivially_copyable(S); // expected-error{{incomplete type 'S' used in type trait expression}}
  __is_trivially_copyable(S[]); // expected-error{{incomplete type 'S' used in type trait expression}}

  __is_trivially_relocatable(S); // expected-error{{incomplete type 'S' used in type trait expression}}
  __is_trivially_relocatable(S[]); // expected-error{{incomplete type 'S' used in type trait expression}}
}
