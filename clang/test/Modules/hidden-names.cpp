// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/hidden-names %s -verify
// expected-no-diagnostics

#include "visible.h"

using namespace NS;

namespace {
  struct X { void f(); };
}

void X::f() {}
