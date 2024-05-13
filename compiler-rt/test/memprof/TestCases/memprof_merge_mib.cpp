// RUN: %clangxx_memprof  %s -o %t

// RUN: %env_memprof_opts=print_text=true:log_path=stdout %run %t | FileCheck %s

#include <sanitizer/memprof_interface.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  for (int i = 0; i < 3; i++) {
    char *x = (char *)malloc(10);
    if (i % 2)
      memset(x, 0, 10);
    else
      memset(x, 2, 10);
    free(x);
  }
  return 0;
}
// We should get one allocation site with alloc_count = loop trip count = 3
// CHECK: Memory allocation stack id = [[ID:[0-9]+]]
// CHECK-NEXT-COUNT-1: alloc_count 3
// CHECK-COUNT-1: Stack for id {{.*}}[[ID]]
// CHECK-NEXT-COUNT-1: memprof_malloc_linux.cpp
// CHECK-NEXT-COUNT-1: memprof_merge_mib.cpp
