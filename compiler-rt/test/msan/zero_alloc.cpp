// RUN: %clang_msan -Wno-alloc-size -fsanitize-recover=memory %s -o %t && not %run %t 2>&1 \
// RUN:     | FileCheck %s --check-prefix=CHECK
// RUN: %clang_msan -Wno-alloc-size -fsanitize-recover=memory -fsanitize-memory-track-origins=1 %s -o %t && not %run %t 2>&1 \
// RUN:     | FileCheck %s --check-prefixes=CHECK,DISCOUNT
// RUN: %clang_msan -Wno-alloc-size -fsanitize-recover=memory -fsanitize-memory-track-origins=2 %s -o %t && not %run %t 2>&1 \
// RUN:     | FileCheck %s --check-prefixes=CHECK,ORIGINS

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  {
    char *p1 = (char *)calloc(1, 0);
    printf("p1 is %p\n", p1);
    printf("Content of p1 is: %d\n", *p1);
    // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
    // CHECK: {{#0 0x.* in main .*zero_alloc.cpp:}}[[@LINE-2]]
    // DISCOUNT,ORIGINS: Uninitialized value is outside of heap allocation
    free(p1);
  }

  {
    char *p2 = (char *)calloc(0, 1);
    printf("p2 is %p\n", p2);
    printf("Content of p2 is: %d\n", *p2);
    // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
    // CHECK: {{#0 0x.* in main .*zero_alloc.cpp:}}[[@LINE-2]]
    // DISCOUNT,ORIGINS: Uninitialized value is outside of heap allocation
    free(p2);
  }

  {
    char *p3 = (char *)malloc(0);
    printf("p3 is %p\n", p3);
    printf("Content of p2 is: %d\n", *p3);
    // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
    // CHECK: {{#0 0x.* in main .*zero_alloc.cpp:}}[[@LINE-2]]
    // DISCOUNT: Uninitialized value was created by a heap allocation
    // ORIGINS: Uninitialized value is outside of heap allocation
    free(p3);
  }

  return 0;
}
