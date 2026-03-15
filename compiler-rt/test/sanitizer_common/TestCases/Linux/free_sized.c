// RUN: %clang -std=c23 -O0 %s -o %t && %run %t
// UNSUPPORTED: asan, hwasan, ubsan

#include <stddef.h>
#include <stdlib.h>

extern void free_sized(void *p, size_t size);

int main() {
  volatile void *p = malloc(64);
  free_sized((void *)p, 64);
  return 0;
}
