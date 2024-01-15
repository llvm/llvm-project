// RUN: %clang_cc1 -std=c++14 -fsyntax-only %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -fexperimental-new-constant-interpreter %s

#include <limits.h>

constexpr unsigned inc() {
  unsigned i = INT_MAX;
  ++i; // should not warn value is outside range
  return i;
}

constexpr unsigned dec() {
  unsigned i = INT_MIN;
  --i; // should not warn value is outside range
  return i;
}
