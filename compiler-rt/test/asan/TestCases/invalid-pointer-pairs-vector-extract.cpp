// REQUIRES: asserts
// RUN: %clangxx_asan -O2 %s -o %t -mllvm -asan-detect-invalid-pointer-pair -fwrapv-pointer
// RUN: %env_asan_opts=detect_invalid_pointer_pairs=1:halt_on_error=0 %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=detect_invalid_pointer_pairs=2:halt_on_error=0 %run %t 2>&1 | FileCheck %s

// UNSUPPORTED: target={{.*}}
// UNSUPPORTED: windows
// UNSUPPORTED: target={{.*solaris.*}}

// XFAIL: *

#include <cstdint>
#include <stdio.h>
#include <stdlib.h>

__attribute__((noinline)) void
span_lengths(const char **begins, const char **ends, int64_t *lengths) {
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in span_lengths
  lengths[0] = ends[0] - begins[0];
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in span_lengths
  lengths[1] = ends[1] - begins[1];
}

int main() {
  char *a = (char *)malloc(16);
  char *b = (char *)malloc(16);
  const char *begins[2] = {a, b};
  const char *ends[2] = {b + 2, a + 10};
  int64_t lengths[2] = {0};
  span_lengths(begins, ends, lengths);
  fprintf(stderr, "lengths: %lld %lld\n", (long long)lengths[0],
          (long long)lengths[1]);
  free(a);
  free(b);
  return 0;
}
