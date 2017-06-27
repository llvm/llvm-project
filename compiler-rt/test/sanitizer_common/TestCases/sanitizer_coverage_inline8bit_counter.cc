// Tests -fsanitize-coverage=inline-8bit-counters
//
// REQUIRES: has_sancovcc,stable-runtime
// UNSUPPORTED: i386-darwin
//
// RUN: %clangxx -O0 %s -fsanitize-coverage=inline-8bit-counters 2>&1

#include <stdio.h>
#include <assert.h>

const char *first_counter;

extern "C"
void __sanitizer_cov_8bit_counters_init(const char *start, const char *end) {
  printf("INIT: %p %p\n", start, end);
  assert(end - start > 1);
  first_counter = start;
}

int main() {
  assert(first_counter);
  assert(*first_counter == 1);
}
