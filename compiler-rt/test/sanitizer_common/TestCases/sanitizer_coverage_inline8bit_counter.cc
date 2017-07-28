// Tests -fsanitize-coverage=inline-8bit-counters,pc-table
//
// REQUIRES: has_sancovcc,stable-runtime
// UNSUPPORTED: i386-darwin
//
// RUN: %clangxx -O0 %s -fsanitize-coverage=inline-8bit-counters,pc-table 2>&1

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

const char *first_counter;

extern "C"
void __sanitizer_cov_8bit_counters_init(const char *start, const char *end) {
  printf("INIT: %p %p\n", start, end);
  assert(end - start > 1);
  first_counter = start;
}

uintptr_t FirstPC;

extern "C" void __sanitizer_cov_pcs_init(const uint8_t *pcs_beg,
                                         const uint8_t *pcs_end) {
  const uintptr_t *B = (const uintptr_t *)pcs_beg;
  const uintptr_t *E = (const uintptr_t *)pcs_end;
  assert(B < E);
  FirstPC = *B;
}


int main() {
  assert(first_counter);
  assert(*first_counter == 1);
  assert(FirstPC == (uintptr_t)&main);
}
