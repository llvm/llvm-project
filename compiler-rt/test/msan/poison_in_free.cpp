// RUN: %clangxx_msan -O0 %s -o %t && not %run %t >%t.out 2>&1
// FileCheck %s <%t.out
// RUN: %clangxx_msan -O0 %s -o %t && env MSAN_OPTIONS=poison_in_free=0 %run %t >%t.out 2>&1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  char *volatile x = (char*)malloc(50 * sizeof(char));
  memset(x, 0, 50);
  free(x);
  return x[25];
  // CHECK: MemorySanitizer: use-of-uninitialized-value
  // CHECK: #0 {{.*}} in main{{.*}}poison_in_free.cpp:[[@LINE-2]]
}
