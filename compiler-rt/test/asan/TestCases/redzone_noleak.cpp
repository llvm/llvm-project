// Test whether pointers into left redzone count memory are reachable.
// If user thread is inside asan allocator code then we may have no
// pointers into user part of memory yet. However we should have a pointer
// into the allocated memory chunk.
//
// RUN: %clangxx_asan  %s -o %t
// RUN: %run %t 2>&1

#include <cstdlib>
#include <stdio.h>
#include <thread>

void *pointers[1000];
void **cur = pointers;

void leak(int n, int offset) {
  printf("%d %d\n", n, offset);
  for (int i = 0; i < 3; ++i)
    *(cur++) = (new int[n]) + offset;
}

int main(int argc, char **argv) {
  for (int n = 1; n < 10000000; n = n * 2) {
    leak(n, 0);
    leak(n, -1);
  }
  return 0;
}
