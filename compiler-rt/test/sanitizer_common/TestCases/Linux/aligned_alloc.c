// RUN: %clang -std=c11 -O0 %s -o %t && %run %t
#include <stdlib.h>
extern void *aligned_alloc (size_t alignment, size_t size);
int main() {
  volatile void *p = aligned_alloc(128, 1024);
  free((void*)p);
  return 0;
}
