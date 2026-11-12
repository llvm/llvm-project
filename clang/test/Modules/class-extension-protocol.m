// RUN: rm -rf %t.cache
// RUN: %clang_cc1 %s -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.cache -I%S/Inputs/class-extension -verify
// expected-no-diagnostics

#import "a-private.h"

int foo(A *X) {
  return X.p0 + X.p1;
}
