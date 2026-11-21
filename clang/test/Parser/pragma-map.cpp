// RUN: %clang_cc1 -x c++ -triple s390x-ibm-zos -fsyntax-only -verify %s
// REQUIRES: systemz-registered-target

#pragma map(f // expected-warning {{expected ',' in '#pragma map'}}
#pragma map(f( // expected-warning {{missing ')' after '#pragma map' - ignoring}}
#pragma map(f(T // expected-warning {{missing ')' after '#pragma map' - ignoring}}
#pragma map(f(T, "def") // expected-warning {{argument list in '#pragma map' - ignoring}} 
  // expected-warning@-1 {{expected ',' in '#pragma map'}}
extern "C" void f(double, double) {
}
