// REQUIRES: arm-target-arch || armv4t-target-arch || armv6m-target-arch
// RUN: %clang_builtins %s %librt -o %t && %run %t
// RUN: %if !armv6m-target-arch %{ \
// RUN:   %clang_builtins -mthumb %s %librt -o %t.thumb && %run %t.thumb \
// RUN: %}

#include <stddef.h>

extern int __aeabi_memcmp(const void *, const void *, size_t);
extern int __aeabi_memcmp4(const void *, const void *, size_t);
extern void __aeabi_memcpy(void *, const void *, size_t);
extern void __aeabi_memcpy4(void *, const void *, size_t);
extern void __aeabi_memmove(void *, const void *, size_t);
extern void __aeabi_memmove4(void *, const void *, size_t);
extern void __aeabi_memset(void *, size_t, int);
extern void __aeabi_memset4(void *, size_t, int);
extern void __aeabi_memclr(void *, size_t);
extern void __aeabi_memclr4(void *, size_t);

typedef unsigned char buffer[8] __attribute__((aligned(4)));

int check(const unsigned char *p, const unsigned char *expected, size_t n) {
  for (size_t i = 0; i != n; ++i)
    if (p[i] != expected[i])
      return 1;
  return 0;
}

int main(void) {
  unsigned char *source = (buffer){0, 1, 2, 3, 4, 5, 6, 7};
  unsigned char *data =
      (buffer){0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
  unsigned char *expected;

  __aeabi_memcpy(data, source, 8);
  if (check(data, source, 8))
    return 1;

  source = (buffer){0, 1, 2, 3, 4, 5, 6, 7};
  data = (buffer){0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
  __aeabi_memcpy4(data, source, 8);
  if (check(data, source, 8))
    return 1;

  data = (buffer){0, 1, 2, 3, 4, 5, 6, 7};
  expected = (buffer){0, 0, 1, 2, 3, 4, 5, 6};
  __aeabi_memmove(data + 1, data, 7);
  if (check(data, expected, 8))
    return 1;

  data = (buffer){0, 1, 2, 3, 4, 5, 6, 7};
  expected = (buffer){0, 1, 2, 3, 0, 1, 2, 3};
  __aeabi_memmove4(data + 4, data, 4);
  if (check(data, expected, 8))
    return 1;

  data = (buffer){0, 1, 2, 3, 4, 5, 6, 7};
  expected = (buffer){0, 1, 0xff, 0xff, 0xff, 0xff, 6, 7};
  __aeabi_memset(data + 2, 4, 0xff);
  if (check(data, expected, 8))
    return 1;

  data = (buffer){0, 1, 2, 3, 4, 5, 6, 7};
  expected = (buffer){0x5a, 0x5a, 0x5a, 0x5a, 4, 5, 6, 7};
  __aeabi_memset4(data, 4, 0x5a);
  if (check(data, expected, 8))
    return 1;

  data = (buffer){0, 1, 2, 3, 4, 5, 6, 7};
  expected = (buffer){0, 1, 0, 0, 0, 0, 6, 7};
  __aeabi_memclr(data + 2, 4);
  if (check(data, expected, 8))
    return 1;

  data = (buffer){1, 2, 3, 4, 5, 6, 7, 8};
  expected = (buffer){0};
  __aeabi_memclr4(data, 8);
  if (check(data, expected, 8))
    return 1;

  data = (buffer){0, 1, 2, 3, 4, 5, 6, 7};
  expected = (buffer){0, 1, 2, 3, 4, 5, 6, 7};
  if (__aeabi_memcmp(data, expected, 8) != 0)
    return 1;
  expected[7] = 8;
  if (__aeabi_memcmp(data, expected, 8) >= 0)
    return 1;
  if (__aeabi_memcmp(expected, data, 8) <= 0)
    return 1;

  data = (buffer){0, 1, 2, 3, 4, 5, 6, 7};
  expected = (buffer){0, 1, 2, 3, 4, 5, 6, 7};
  if (__aeabi_memcmp4(data, expected, 8) != 0)
    return 1;
  expected[7] = 8;
  if (__aeabi_memcmp4(data, expected, 8) >= 0)
    return 1;
  if (__aeabi_memcmp4(expected, data, 8) <= 0)
    return 1;

  return 0;
}
