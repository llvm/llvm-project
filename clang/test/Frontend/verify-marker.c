// RUN: %clang_cc1 -verify %s

#include "verify-marker.h" // expected-error@#1 {{unknown type name 'unknown_type'}}

int x = 1; // #a
int x = 2; // #b
// expected-error@#b {{redefinition of 'x'}}
// expected-note@#a {{previous}}

// expected-error@#unknown {{}}  expected-error {{use of undefined marker '#unknown'}}

// This is OK: there's no problem with a source file containing what looks like
// a duplicate definition of a marker if that marker is never used.
// #foo
// #foo

// #bar  expected-note {{ambiguous marker '#bar' is defined here}}
// #bar  expected-note {{ambiguous marker '#bar' is defined here}}
// expected-error@#bar 0-1{{oops}}  expected-error{{reference to marker '#bar' is ambiguous}}

// expected-error@#forward_ref {{undeclared identifier 'future'}}
int y = future; // #forward_ref
