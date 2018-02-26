// RUN: %clangxx -fsanitize=integer -g0 %s -o %t

// Suppression by symbol name (unsigned-integer-overflow:do_overflow below)
// requires the compiler-rt runtime to be able to symbolize stack addresses.
// REQUIRES: can-symbolize
// UNSUPPORTED: android

// Fails without any suppression.
// RUN: %env_ubsan_opts=halt_on_error=1 not %run %t 2>&1 | FileCheck %s

// RUN: echo "signed-integer-overflow:%t" > %t.wrong-supp
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.wrong-supp"' not %run %t 2>&1 | FileCheck %s

// RUN: echo "unsigned-integer-overflow:do_overflow" > %t.func-supp
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.func-supp"' %run %t
// RUN: echo "unsigned-integer-overflow:%t" > %t.module-supp
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.module-supp"' %run %t

// Note: file-level suppressions should work even without debug info.
// RUN: echo "unsigned-integer-overflow:%s" > %t.file-supp
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.file-supp"' %run %t

// Suppressions don't work for unrecoverable kinds.
// RUN: %clangxx -fsanitize=integer -fno-sanitize-recover=integer %s -o %t-norecover
// RUN: %env_ubsan_opts=halt_on_error=1:suppressions='"%t.module-supp"' not %run %t-norecover 2>&1 | FileCheck %s

#include <stdint.h>

extern "C" void do_overflow() {
  (void)(uint64_t(10000000000000000000ull) + uint64_t(9000000000000000000ull));
  // CHECK: runtime error: unsigned integer overflow
}

int main() {
  do_overflow();
  return 0;
}
