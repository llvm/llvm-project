// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'struct tm;' > %t/a.h
// RUN: echo 'struct X {}; void foo(struct tm*);' > %t/b.h
// RUN: echo 'module X { module a { header "a.h" } module b { header "b.h" } }' > %t/x.modulemap
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -x c++ -fmodule-map-file=%t/x.modulemap %s -I%t -verify -std=c++11
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -x c++ -fmodule-map-file=%t/x.modulemap %s -I%t -verify -fmodules-local-submodule-visibility -std=c++11

#include "a.h"

using ::tm;

struct A {
  // This use of 'struct X' makes the declaration (but not definition) of X visible.
  virtual void f(struct X *p);
};

namespace N {
  struct B : A {
    void f(struct X *q) override;
  };
}

X x; // expected-error {{'X' must be defined before it is used}}
// expected-note@b.h:1 {{here}}
