// RUN: %clang_cc1 -verify -chain-include %s %s

// PR 14044
#ifndef PASS1
#define PASS1
class S {
  void f(struct Test);
};
#else
::Tesy *p;  // expected-error {{did you mean 'Test'}}
            // expected-note@-4 {{'Test' declared here}}
#endif
