// RUN: %clang_cc1 -triple s390x-ibm-zos -fsyntax-only -verify %s

int x;

#pragma export x  // expected-warning {{missing '(' after '#pragma export' - ignoring}}
#pragma export  // expected-warning {{missing '(' after '#pragma export' - ignoring}}
#pragma export(  // expected-warning {{expected identifier in '#pragma export' - ignored}}
#pragma export(x  // expected-warning {{missing ')' after '#pragma export' - ignoring}}
#pragma export(::x) // expected-warning {{expected identifier in '#pragma export' - ignored}}
#pragma export(x)

void f() {
}

#pragma export(f()) // expected-warning {{missing ')' after '#pragma export' - ignoring}}
