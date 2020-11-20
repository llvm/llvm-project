// RUN: %clangxx_memprof  %s -o %t

// RUN: %env_memprof_opts=log_path=stdout %run %t | FileCheck %s

#include <sanitizer/memprof_interface.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  __memprof_profile_dump();
  x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  return 0;
}
// We should get 2 rounds of profile info, one from the explicit dump request,
// and one at exit.
// CHECK: Memory allocation stack id
// CHECK: Stack for id
// CHECK: Memory allocation stack id
// CHECK: Stack for id
