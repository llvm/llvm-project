// RUN: %clang_cc1 -x c++ -triple s390x-ibm-zos -fsyntax-only -verify %s

extern int i;
#pragma export(:: // expected-warning {{expected identifier in '#pragma export' - ignored}}
#pragma export(::) // expected-warning {{expected identifier in '#pragma export' - ignored}}
#pragma export(::i)

struct S {
  static int i;
};
#pragma export(S:: // expected-warning {{expected identifier in '#pragma export' - ignored}}
#pragma export(S::i // expected-warning {{missing ')' after '#pragma export' - ignoring}}
#pragma export(S::i)

void f(int);
void f(double, double);
#pragma export(f( // expected-warning {{expected identifier in '#pragma export' - ignored}}
#pragma export(f() // expected-warning {{missing ')' after '#pragma export' - ignoring}}
#pragma export(f(int) // expected-warning {{missing ')' after '#pragma export' - ignoring}}
#pragma export(f(double,) // expected-warning {{missing ')' after '#pragma export' - ignoring}}
#pragma export(f(double,double))
