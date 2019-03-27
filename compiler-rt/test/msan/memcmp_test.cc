// RUN: %clangxx_msan -O0 -g %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: MSAN_OPTIONS=intercept_memcmp=0 %run %t

#include <string.h>
#include <stdio.h>
int main(int argc, char **argv) {
  char a1[4];
  char a2[4];
  for (int i = 0; i < argc * 3; i++)
    a2[i] = a1[i] = i;
  int res = memcmp(a1, a2, 4);
  if (!res)
    printf("equals");
  return 0;
  // CHECK: Uninitialized bytes in MemcmpInterceptorCommon at offset 3
  // CHECK: MemorySanitizer: use-of-uninitialized-value
}
