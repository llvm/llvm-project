// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace a { b c ( a:c::
// expected-error@-1 {{unknown type name 'b'}}
// expected-error@-2 {{unexpected ':' in nested name specifier; did you mean '::'?}}
// expected-error@-3 {{no member named 'c' in namespace 'a'}}
// expected-error@-4 {{expected ';' after top level declarator}}
// expected-note@-5 {{to match this '{'}}
// expected-error@+1 {{expected unqualified-id}} \
// expected-error@+1 {{expected '}'}}
