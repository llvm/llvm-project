// Tests -fsanitize-coverage=inline-bool-flag,pc-table
//
// REQUIRES: has_sancovcc,stable-runtime
// UNSUPPORTED: i386-darwin
//
// RUN: %clangxx -O0 %s -fsanitize-coverage=inline-bool-flag,pc-table -o %t
// RUN: %run %t 2>&1 | FileCheck %s
// XFAIL: tsan

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

const bool *first_flag;

extern "C" void __sanitizer_cov_bool_flag_init(const bool *start,
                                               const bool *end) {
  printf("INIT: %p %p\n", start, end);
  assert(end - start > 1);
  first_flag = start;
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
  assert(first_flag);
  assert(*first_flag);
  assert(FirstPC == (uintptr_t)&main);
  assert(FirstPCFlag == 1);
  fprintf(stderr, "PASS\n");
  // CHECK: PASS
}
