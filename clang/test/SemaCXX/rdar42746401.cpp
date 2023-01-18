// RUN: %clang_cc1 -fsyntax-only -verify %s

template <int>
class b;
class c; // expected-note{{forward declaration}}

::b<0> struct c::d // expected-error{{incomplete type}} expected-error{{cannot combine}} expected-error{{expected unqualified-id}}
