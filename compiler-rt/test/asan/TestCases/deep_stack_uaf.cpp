// Check that we can store lots of stack frames if asked to.

// RUN: %clangxx_asan -O0 %s -o %t 2>&1
// RUN: %env_asan_opts=malloc_context_size=120:redzone=512 not %run %t 2>&1 | FileCheck %s
// REQUIRES: stable-runtime
#include <stdlib.h>
#include <stdio.h>

template <int depth>
struct DeepFree {
  static void free(char *x) {
    DeepFree<depth - 1>::free(x);
  }
};

template<>
struct DeepFree<0> {
  static void free(char *x) {
    ::free(x);
  }
};

int main() {
  char *x = (char*)malloc(10);
  // deep_free(x);
  DeepFree<200>::free(x);
  return x[5];
  // CHECK: {{.*ERROR: AddressSanitizer: heap-use-after-free on address}}
  // The libcxxrt demangling procedure on FreeBSD 9.2 incorrectly appends
  // extra 'E' characters to the end of template arguments; see:
  // https://bugs.freebsd.org/bugzilla/show_bug.cgi?id=192115
  // CHECK: {{DeepFree<36>|DeepFree<36E>}}
  // CHECK: {{DeepFree<98>|DeepFree<98E>}}
  // CHECK: {{DeepFree<115>|DeepFree<115E>}}
}
