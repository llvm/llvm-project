// Test that we use the suppressions from __ubsan_default_suppressions.
// RUN: %clangxx -fsanitize=integer -fsanitize-recover=integer -g0 %s -o %t
// RUN: %env_ubsan_opts=halt_on_error=1 %run %t

// Suppression by symbol name requires the compiler-rt runtime to be able to
// symbolize stack addresses.
// REQUIRES: can-symbolize
// UNSUPPORTED: android

#include <sanitizer/ubsan_interface.h>
#include <stdint.h>

extern "C" const char *__ubsan_default_suppressions() {
  return "unsigned-integer-overflow:do_overflow";
}

extern "C" void do_overflow() {
  (void)(uint64_t(10000000000000000000ull) + uint64_t(9000000000000000000ull));
}

int main() {
  do_overflow();
  return 0;
}
