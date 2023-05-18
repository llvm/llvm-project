// RUN: %clangxx -fsanitize=signed-integer-overflow -w %s -O3 -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=RECOVER

// RUN: %clangxx -fsanitize=signed-integer-overflow -fno-sanitize-recover=signed-integer-overflow -w %s -O3 -o %t.abort
// RUN: not %run %t.abort 2>&1 | FileCheck %s --check-prefix=ABORT

#include <limits.h>
#include <stdlib.h>

int main() {
  // ABORT: abs.cpp:[[#@LINE+3]]:17: runtime error: negation of -[[#]] cannot be represented in type 'int'; cast to an unsigned type to negate this value to itself
  // RECOVER: abs.cpp:[[#@LINE+2]]:17: runtime error: negation of -[[#]] cannot be represented in type 'int'; cast to an unsigned type to negate this value to itself
  // RECOVER: abs.cpp:[[#@LINE+2]]:7: runtime error: negation of -[[#]] cannot be represented in type 'int'; cast to an unsigned type to negate this value to itself
  __builtin_abs(INT_MIN);
  abs(INT_MIN);

  // RECOVER: abs.cpp:[[#@LINE+2]]:18: runtime error: negation of -[[#]] cannot be represented in type 'long'; cast to an unsigned type to negate this value to itself
  // RECOVER: abs.cpp:[[#@LINE+2]]:8: runtime error: negation of -[[#]] cannot be represented in type 'long'; cast to an unsigned type to negate this value to itself
  __builtin_labs(LONG_MIN);
  labs(LONG_MIN);

  // RECOVER: abs.cpp:[[#@LINE+2]]:19: runtime error: negation of -[[#]] cannot be represented in type 'long long'; cast to an unsigned type to negate this value to itself
  // RECOVER: abs.cpp:[[#@LINE+2]]:9: runtime error: negation of -[[#]] cannot be represented in type 'long long'; cast to an unsigned type to negate this value to itself
  __builtin_llabs(LLONG_MIN);
  llabs(LLONG_MIN);

  return 0;
}
