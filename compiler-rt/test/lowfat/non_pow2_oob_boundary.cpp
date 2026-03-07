// RUN: %clangxx_lowfat -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat_safe -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s

// GEP-level instrumentation test: p + 48 is one-past a 48-byte object and must
// be reported before the pointer escapes to sink().
//
// REQUIRES: lowfat-custom-config

#include <stdlib.h>

// noinline keeps this as a cross-function pointer-escape case.
__attribute__((noinline))
static void sink(volatile char *q) { *q = 'x'; }

int main() {
  char *p = (char *)malloc(48);
  if (!p) return 1;

  // GEP check: result == End, so this is out of bounds.
  // CHECK: LOWFAT ERROR: out-of-bounds error detected!
  sink(p + 48);

  free(p);
  return 0;
}

