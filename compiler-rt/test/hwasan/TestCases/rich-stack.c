// Test how stack frames are reported (not fully implemented yet).
// RUN: %clang_hwasan %s -o %t
// RUN: not %run %t 3 2 -1 2>&1 | FileCheck %s --check-prefix=R321
// REQUIRES: stable-runtime, pointer-tagging

#include <stdint.h>
#include <stdlib.h>
void USE(void *x) { // pretend_to_do_something(void *x)
  __asm__ __volatile__("" : : "r" (x) : "memory");
}
void USE2(void *a, void *b) { USE(a); USE(b); }
void USE4(void *a, void *b, void *c, void *d) { USE2(a, b); USE2(c, d); }

void BAR(int depth, int err_depth, int offset);

uint64_t *leaked_ptr;

void FOO(int depth, int err_depth, int offset) {
  uint8_t v1;
  uint16_t v2;
  uint32_t v4;
  uint64_t v8;
  uint64_t v16[2];
  uint64_t v32[4];
  uint64_t v48[3];
  USE4(&v1, &v2, &v4, &v8);  USE4(&v16, &v32, &v48, 0);
  leaked_ptr = &v16[0];
  if (depth)
    BAR(depth - 1, err_depth, offset);

  if (err_depth == depth)
    v16[offset] = 0;  // maybe OOB.
  if (err_depth == -depth)
    leaked_ptr[offset] = 0; // maybe UAR.
  USE(&v16);
}

void BAR(int depth, int err_depth, int offset) {
  uint64_t x16[2];
  uint64_t x32[4];
  USE2(&x16, &x32);
  leaked_ptr = &x16[0];
  if (depth)
    FOO(depth - 1, err_depth, offset);
  if (err_depth == depth)
    x16[offset] = 0;  // maybe OOB
  if (err_depth == -depth)
    leaked_ptr[offset] = 0;  // maybe UAR
  USE(&x16);
}


int main(int argc, char **argv) {
  if (argc != 4) return -1;
  int depth = atoi(argv[1]);
  int err_depth = atoi(argv[2]);
  int offset = atoi(argv[3]);
  FOO(depth, err_depth, offset);
  return 0;
}

// R321: HWAddressSanitizer: tag-mismatch
// R321-NEXT: WRITE of size 8
// R321: in BAR
// R321-NEXT: in FOO
// R321-NEXT: in main
// R321: is located in stack of thread T0
