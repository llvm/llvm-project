// Tests -fsanitize-coverage=inline-8bit-counters,pc-table
//
// REQUIRES: has_sancovcc,stable-runtime
// UNSUPPORTED: i386-darwin
//
// RUN: %clangxx -O0 %s -fsanitize-coverage=inline-8bit-counters,pc-table -o %t
// RUN: %run %t 2>&1 | FileCheck %s
// XFAIL: tsan

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
uintptr_t FirstPCFlag;

extern "C" void __sanitizer_cov_pcs_init(const uintptr_t *pcs_beg,
                                         const uintptr_t *pcs_end) {
  const uintptr_t *B = (const uintptr_t *)pcs_beg;
  const uintptr_t *E = (const uintptr_t *)pcs_end;
  assert(B + 1 < E);
  FirstPC = B[0];
  FirstPCFlag = B[1];
}


int main() {
  assert(first_counter);
  assert(*first_counter == 1);
  assert(FirstPC == (uintptr_t)&main);
  assert(FirstPCFlag == 1);
  fprintf(stderr, "PASS\n");
  // CHECK: PASS
}
