// RUN: %clang_cc1 -fsyntax-only %s -chain-include %s -Wuninitialized -Wunused -verify

// Make sure there is no crash.

#ifndef HEADER
#define HEADER

#include "non-existent-header.h"

class A {
public:
  ~A();
};

class ForwardCls;
struct B {
  ForwardCls f;
  A a;
};

#else

static void test() {
  int x; // expected-warning {{unused}}
  B b;
}

#endif
