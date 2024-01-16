// REQUIRES: linux, aarch64-target-arch, aarch64-sme-available
// RUN: %clangxx_builtins %s %librt -o %t && %run %t

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define N 16

extern "C" {
void *__arm_sc_memcpy(void *, const void *, size_t);
void *__arm_sc_memset(void *, int, size_t);
void *__arm_sc_memmove(void *, const void *, size_t);
void *__arm_sc_memchr(const void *, int, size_t);
}

class MemoryArea {

  uint8_t dst[N], src[N];

  int sum_and_reset_dst(uint8_t *dest, int n, int j) {
    int t = 0;
    for (int i = 0; i < n; i++) {
      t += dest[i];
    }
    for (int i = 0; i < j; i++) {
      dst[i] = i + 1;
    }
    return t;
  }

public:
  MemoryArea() {
    for (int i = 0; i < N; i++) {
      src[i] = i * 2;
      dst[i] = i + 1;
    }
  }

  // Test correctness of memcpy
  void test_memcpy() {
    for (int i = 0; i < 8; i++) {
      int t[2];
      if (!__arm_sc_memcpy(dst, src, i))
        abort();
      t[0] = sum_and_reset_dst(dst, N, i);
      memcpy(dst, src, i);
      t[1] = sum_and_reset_dst(dst, N, i);
      if (t[0] != t[1])
        abort();
    }
  }

  // Test correctness of memset
  void test_memset() {
    for (int i = 0; i < 8; i++) {
      int t[2];
      if (!__arm_sc_memset(dst, src[i], i))
        abort();
      t[0] = sum_and_reset_dst(dst, N, i);
      __arm_sc_memset(dst, src[i], i);
      t[1] = sum_and_reset_dst(dst, N, i);
      if (t[0] != t[1])
        abort();
    }
  }

  // Test correctness of memchr
  void test_memchr() {
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        uint8_t *t[2];
        t[0] = (uint8_t *)__arm_sc_memchr(src, src[j], i);
        t[1] = (uint8_t *)__arm_sc_memchr(src, src[j], i);
        if (t[0] != t[1])
          abort();
      }
    }
  }

  // Test correctness for memmove
  void test_memmove() {
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        int t[2];
        if (!__arm_sc_memmove(&dst[8 - j], &dst[j], i))
          abort();
        t[0] = sum_and_reset_dst(dst, N, 16);
        __arm_sc_memmove(&dst[8 - j], &dst[j], i);
        t[1] = sum_and_reset_dst(dst, N, 16);
        if (t[0] != t[1])
          abort();
      }
    }
  }
};

int main() {

  MemoryArea MA = MemoryArea();

  MA.test_memcpy();
  MA.test_memset();
  MA.test_memchr();
  MA.test_memmove();

  return 0;
}
