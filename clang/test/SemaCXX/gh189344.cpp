// RUN: %clang_cc1 -fsyntax-only -verify %s

template <union { } alignas ( union  a { c ) ( a :: ) ( :: (b
// expected-error@-1 {{cannot be defined in a type specifier}}
// expected-error@-2 {{'a' cannot be defined in a type specifier}}
// expected-error@-3 {{unknown type name 'b'}}
// expected-error@-4 2{{parameter declarator cannot be qualified}}
// expected-error@-5 {{a type specifier is required for all declarations}}
// expected-note@-6 2{{to match this '('}}
// expected-error@* 3{{expected unqualified-id}}
// expected-error@* 2{{expected ')'}}
// expected-error@* {{expected ',' or '>' in template-parameter-list}}
