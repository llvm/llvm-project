// RUN: %clang_msan -Wno-alloc-size -fsanitize-recover=memory %s -o %t && not %run %t 2>&1 | FileCheck %s

// MSan doesn't catch this because internally it translates 0-byte allocations
// into 1-byte
// XFAIL: *

#include <malloc.h>
#include <stdio.h>

int main(int argc, char **argv) {
  {
    char *p1 = (char *)calloc(1, 0);
    printf("p1 is %p\n", p1);
    printf("Content of p1 is: %d\n", *p1);
    // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
    // CHECK: {{#0 0x.* in main .*zero_alloc.cpp:}}[[@LINE-2]]
    free(p1);
  }

  {
    char *p2 = (char *)calloc(0, 1);
    printf("p2 is %p\n", p2);
    printf("Content of p2 is: %d\n", *p2);
    // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
    // CHECK: {{#0 0x.* in main .*zero_alloc.cpp:}}[[@LINE-2]]
    free(p2);
  }

  {
    char *p3 = (char *)malloc(0);
    printf("p3 is %p\n", p3);
    printf("Content of p2 is: %d\n", *p3);
    // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
    // CHECK: {{#0 0x.* in main .*zero_alloc.cpp:}}[[@LINE-2]]
    free(p3);
  }

  return 0;
}
