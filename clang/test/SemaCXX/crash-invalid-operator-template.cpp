// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef(( operator | <> ) ) ( class { private: }); 
// expected-error@-1 {{cannot be defined in a parameter type}}
// expected-error@-2 {{type specifier is required}}
// expected-error@-3 {{typedef name must be an identifier}}

