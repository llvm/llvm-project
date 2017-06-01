// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-%short-stack %s < %t.out
// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -O2 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck --check-prefix=CHECK --check-prefix=CHECK-%short-stack %s < %t.out

// This is a regression test: there used to be broken "stored to memory at"
// stacks with
//   in __msan_memcpy
//   in __msan::MsanReallocate
// and nothing below that.

#include <stdlib.h>
int main(int argc, char **argv) {
  char *p = (char *)malloc(100);
  p = (char *)realloc(p, 10000);
  char x = p[50];
  free(p);
  return x;

// CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
// CHECK:   {{#0 0x.* in main .*realloc-large-origin.cc:}}[[@LINE-3]]

// CHECK:  Uninitialized value was stored to memory at
// CHECK-FULL-STACK:   {{#0 0x.* in .*realloc}}
// CHECK-FULL-STACK:   {{#1 0x.* in main .*realloc-large-origin.cc:}}[[@LINE-10]]
// CHECK-SHORT-STACK:   {{#0 0x.* in .*realloc}}

// CHECK:   Uninitialized value was created by a heap allocation
// CHECK:   {{#0 0x.* in .*malloc}}
// CHECK:   {{#1 0x.* in main .*realloc-large-origin.cc:}}[[@LINE-16]]
}
