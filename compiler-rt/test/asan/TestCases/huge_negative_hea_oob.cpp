// RUN: %clangxx_asan %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O %s -o %t && not %run %t 2>&1 | FileCheck %s
// Check that we can find huge buffer overflows before.
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char*)malloc(1 << 20);
  memset(x, 0, 10);
  int res = x[-argc * 4000];  // BOOOM
  // CHECK: is located 4000 bytes before
  free(x);
  return res;
}
