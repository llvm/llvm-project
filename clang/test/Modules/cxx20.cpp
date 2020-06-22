// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -std=c++20 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/cxx20 %s -verify -fno-modules-error-recovery

// expected-no-diagnostics

#include "decls.h"

namespace StructuredBinding {
  struct R { int x, y; };
  static auto [a, b] = R();
}
