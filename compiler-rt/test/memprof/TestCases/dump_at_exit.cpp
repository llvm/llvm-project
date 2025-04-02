// RUN: %clangxx_memprof %s -o %t

// RUN: %env_memprof_opts=print_text=true:log_path=stdout:dump_at_exit=false %run %t | count 0
// RUN: %env_memprof_opts=print_text=true:log_path=stdout:dump_at_exit=true %run %t | FileCheck %s

#include <stdlib.h>
#include <string.h>

int main() {
  char *x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  return 0;
}

// CHECK: Recorded MIBs
