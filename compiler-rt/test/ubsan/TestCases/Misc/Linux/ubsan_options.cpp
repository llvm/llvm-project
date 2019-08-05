// RUN: %clangxx -fsanitize=integer -fsanitize-recover=integer %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// __ubsan_default_options() doesn't work on Darwin.
// XFAIL: darwin

#include <stdint.h>

extern "C" const char *__ubsan_default_options() {
  return "halt_on_error=1";
}

int main() {
  (void)(uint64_t(10000000000000000000ull) + uint64_t(9000000000000000000ull));
  // CHECK: ubsan_options.cpp:[[@LINE-1]]:44: runtime error: unsigned integer overflow
  return 0;
}

