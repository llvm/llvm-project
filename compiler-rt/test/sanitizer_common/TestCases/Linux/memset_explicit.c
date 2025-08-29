// RUN: %clang -std=c23 -O0 %s -o %t && %run %t
// UNSUPPORTED: asan, lsan, hwasan, ubsan

#include <stddef.h>
#include <stdlib.h>

extern void *memset_explicit(void *p, int value, size_t size);

int main() {
  char secbuffer[64];
  (void)memset_explicit(secbuffer, 0, sizeof(secbuffer));
  return 0;
}
