// RUN: %clang_cc1 -fsyntax-only -verify %s

// This invalid declaration used to call infinite recursion in linkage
// calculation for enum as a function argument.
inline foo(A)(enum E;
// expected-error@-1 {{unknown type name 'foo'}}
// expected-error@-2 {{ISO C++ forbids forward references to 'enum' types}}
// expected-error@-3 {{expected ')'}}
// expected-note@-4 {{to match this '('}}
