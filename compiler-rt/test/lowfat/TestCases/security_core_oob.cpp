// RUN: %clangxx_lowfat -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-READ
// RUN: %clangxx_lowfat -O0 %s -DWRITE -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-WRITE
// RUN: %clangxx_lowfat -O0 %s -DUNDERFLOW -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-UNDERFLOW

#include <stdlib.h>
#include <stdio.h>

int main() {
  char *p = (char *)malloc(16);
  if (!p) return 1;

#ifdef UNDERFLOW
  // CHECK-UNDERFLOW: LOWFAT ERROR: out-of-bounds error detected!
  // CHECK-UNDERFLOW: operation = read
  char c = p[-1];
  (void)c;
#elif defined(WRITE)
  // CHECK-WRITE: LOWFAT ERROR: out-of-bounds error detected!
  // Note: GEP-level instrumentation fires before the store, and it currently
  // always reports 'read' (0).
  // CHECK-WRITE: operation = read
  p[16] = 'x';
#else
  // CHECK-READ: LOWFAT ERROR: out-of-bounds error detected!
  // CHECK-READ: operation = read
  char c = p[16];
  (void)c;
#endif

  free(p);
  return 0;
}
