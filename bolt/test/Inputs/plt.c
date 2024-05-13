#include "stub.h"

void *(*memcpy_p)(void *dest, const void *src, unsigned long n);
void *(*memset_p)(void *dest, int c, unsigned long n);

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

  printf("Test completed\n");
}
