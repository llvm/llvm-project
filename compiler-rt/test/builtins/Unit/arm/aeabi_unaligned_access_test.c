// REQUIRES: arm-target-arch || armv6m-target-arch
// RUN: %clang_builtins %s %librt -o %t && %run %t

#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern int __aeabi_uread4(void *);
extern int __aeabi_uwrite4(int, void *);
extern long long __aeabi_uread8(void *);
extern long long __aeabi_uwrite8(long long, void *);

int test_unaligned(void) {
  long long target8;
  int target4;
  const char source[] = "abcdefghijklmno";
  static char dest1[_Countof(source)], dest2[_Countof(source)];
  int i, j;

  for (i = 0; i < 7; i++) {
    memcpy(&target8, source + i, 8);
    if (__aeabi_uread8(source + i) != target8) {
      printf("error in __aeabi_uread8 => output = %llx, expected %llx\n",
             __aeabi_uread8(source + i), target8);
      return 1;
    }

    memcpy(dest1, source, _Countof(source));
    memcpy(dest2, source, _Countof(source));
    target8 = 0x4142434445464748ULL;
    if (__aeabi_uwrite8(target8, dest1 + i) != target8) {
      printf("error in __aeabi_uwrite8 => output = %llx, expected %llx\n",
             __aeabi_uwrite8(target8, dest1 + i), target8);
      return 1;
    }
    memcpy(dest2 + i, &target8, 8);
    if (memcmp(dest1, dest2, _Countof(source)) != 0) {
      int pos = -1;
      printf("error in __aeabi_uwrite8: memcmp failed: buffers differ!\n");
      for (int j = 0; j < 8; ++j) {
        if (dest1[j] != dest2[j]) {
          pos = j;
          break;
        }
      }
      printf("error: 8-byte write mismatch at offset %d\n", pos);
      return 1;
    }

    memcpy(&target4, source + i, 4);
    if (__aeabi_uread4(source + i) != target4) {
      printf("error in __aeabi_uread4 => output = %x, expected %x\n",
             __aeabi_uread4(source + i), target4);
      return 1;
    }

    memcpy(dest1, source, _Countof(source));
    memcpy(dest2, source, _Countof(source));
    target4 = 0x414243444;
    if (__aeabi_uwrite4(target4, dest1 + i) != target4) {
      printf("error in __aeabi_uwrite4 => output = %x, expected %x\n",
             __aeabi_uwrite4(target4, dest1 + i), target4);
      return 1;
    }
    memcpy(dest2 + i, &target4, 4);
    if (memcmp(dest1, dest2, _Countof(source)) != 0) {
      int pos = -1;
      printf("error in __aeabi_uwrite4: memcmp failed: buffers differ!\n");
      for (int j = 0; j < 4; ++j) {
        if (dest1[j] != dest2[j]) {
          pos = j;
          break;
        }
      }
      printf("error: 4-byte write mismatch at offset %d\n", pos);
      return 1;
    }
  }
  return 0;
}

int main() {
  if (test_unaligned())
    return 1;
  return 0;
}
