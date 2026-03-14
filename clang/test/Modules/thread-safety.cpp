// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fimplicit-module-maps \
// RUN:            -I%S/Inputs/thread-safety -std=c++11 -Wthread-safety \
// RUN:            -verify %s
//
// expected-no-diagnostics

#include "b.h"
#include "c.h"

bool g();
void X::f() {
  m.lock();
  if (g())
    m.unlock();
  else
    unlock(*this);
}
