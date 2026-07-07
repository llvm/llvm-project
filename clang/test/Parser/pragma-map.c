// RUN: %clang_cc1 -triple s390x-ibm-zos -fsyntax-only -verify %s
// REQUIRES: systemz-registered-target

int x;
int x1;

#pragma map x  // expected-warning {{missing '(' after '#pragma map' - ignoring}}
#pragma map  // expected-warning {{missing '(' after '#pragma map' - ignoring}}
#pragma map(  // expected-warning {{expected identifier in '#pragma map' - ignored}}
#pragma map(x  // expected-warning {{expected ',' in '#pragma map'}}
#pragma map(x,  // expected-warning {{expected string literal in '#pragma map'}}
#pragma map(x,"foo"  // expected-warning {{missing ')' after '#pragma map'}}
#pragma map(::x) // expected-warning {{expected identifier in '#pragma map' - ignored}}
#pragma map(x, "abc")
#pragma map(x1, "abc") blah // expected-warning {{extra tokens at end of '#pragma map'}}
