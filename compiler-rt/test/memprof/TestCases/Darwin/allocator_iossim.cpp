// Test that memprof allocator interception works on iOS simulator.
// This exercises the SizeClassAllocator32 path on iOS devices (where
// SANITIZER_CAN_USE_ALLOCATOR64 is 0) and the SizeClassAllocator64 path
// on the simulator (where SANITIZER_CAN_USE_ALLOCATOR64 is 1).

// REQUIRES: iossim

// RUN: %clangxx_memprof -O0 %s -o %t
// RUN: %env_memprof_opts=print_text=true:log_path=stderr %run %t 2>%t.out
// RUN: FileCheck %s < %t.out

// CHECK: Memory allocation stack id
// CHECK: alloc_count

#include <cstdlib>
#include <cstring>

int main() {
  // Test malloc/free interception.
  int *p = (int *)malloc(10 * sizeof(int));
  for (int i = 0; i < 10; i++)
    p[i] = i;
  free(p);

  // Test calloc interception.
  int *q = (int *)calloc(10, sizeof(int));
  for (int i = 0; i < 10; i++)
    q[i] = i;
  free(q);

  // Test realloc interception.
  char *r = (char *)malloc(10);
  memset(r, 'a', 10);
  r = (char *)realloc(r, 20);
  memset(r, 'b', 20);
  free(r);

  // Test C++ new/delete interception.
  int *s = new int[10];
  for (int i = 0; i < 10; i++)
    s[i] = i;
  delete[] s;

  // Test posix_memalign interception.
  void *t;
  int ret = posix_memalign(&t, 64, 128);
  if (ret == 0) {
    memset(t, 0, 128);
    free(t);
  }

  return 0;
}
