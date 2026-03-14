// RUN: %clangxx_asan -O0 %s -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && %run %t 2>&1 | FileCheck %s

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  char *p;
  int res = asprintf(&p, "%d", argc);
  fprintf(stderr, "x%d %sx\n", res, p);
  // CHECK: x1 1x
  free(p);
  fprintf(stderr, "DONE\n");
  // CHECK: DONE
  return 0;
}
