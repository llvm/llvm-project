// This test checks that the pointers to PLT are properly updated.

// RUN: %clang %cflags %s -fuse-ld=lld \
// RUN:    -o %t.exe -Wl,-q
// RUN: llvm-bolt %t.exe -o %t.bolt.exe -use-old-text=0 -lite=0
// RUN: %t.bolt.exe

#include <string.h>

void *(*memcpy_p)(void *dest, const void *src, size_t n);
void *(*memset_p)(void *dest, int c, size_t n);

int main() {
  int a = 0xdeadbeef, b = 0;

  memcpy_p = memcpy;
  memcpy_p(&b, &a, sizeof(b));
  if (b != 0xdeadbeef)
    return 1;

  memset_p = memset;
  memset_p(&a, 0, sizeof(a));
  if (a != 0)
    return 1;
}
