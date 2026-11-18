// RUN: %clangxx_memprof -O0 %s -o %t
// %env_memprof_opts=print_text=true:log_path=stdout %run %t | FileCheck %s

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
  char *p = strdup("memccpy");
  char *d = (char *)malloc(5);
  void *r = memccpy(d, p, 'c', 8);
  int cmp = memcmp(r, "memc", 4);
  free(d);
  free(p);
  return cmp;
}
// CHECK: Memory allocation stack id
