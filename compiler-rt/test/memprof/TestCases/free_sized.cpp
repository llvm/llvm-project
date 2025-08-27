// RUN: %clangxx_memprof %s -o %t

// RUN: %env_memprof_opts=print_text=true:log_path=stdout %run %t | FileCheck %s

#include <stdlib.h>
#include <unistd.h>

extern "C" {
void free_sized(void *ptr, size_t size);
void free_aligned_sized(void *ptr, size_t alignement, size_t size);
}

int main() {
  void *p = aligned_alloc(16, 32);
  free_aligned_sized(p, 16, 32);
  p = malloc(10);
  free_sized(p, 10);
  return 0;
}
// CHECK: Memory allocation stack id
// CHECK: Memory allocation stack id
