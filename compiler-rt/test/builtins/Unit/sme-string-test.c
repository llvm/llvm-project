// REQUIRES: linux, aarch64-target-arch, sme-available
// RUN: %clang_builtins %s %librt -o %t && %run %t

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define N 1024

static uint8_t dst[N], src[N];

extern void *__arm_sc_memcpy(void *, const void *, size_t);
extern void *__arm_sc_memset(void *, int, size_t);
extern void *__arm_sc_memmove(void *, const void *, size_t);
extern void *__arm_sc_memchr(const void *, int, size_t);

void init(void) {
  for (int i = 0; i < N; i++) {
    src[i] = i * 2;
    dst[i] = i + 1;
  }
}

void reinit_dst(int n) {
  for (int i = 0; i < n; i++) {
    dst[i] = i + 1;
  }
}

int sum(uint8_t *dest, int n) {
  int t = 0;
  for (int i = 0; i < n; i++) {
    t += dest[i];
  }
  return t;
}

int main() {

  init();

  // Test correctness of memcpy
  for (int i = 0; i < 67; i++) {
    int t[2];
    if (!__arm_sc_memcpy(dst, src, i))
      abort();
    t[0] = sum(dst, N);
    reinit_dst(i);
    memcpy(dst, src, i);
    t[1] = sum(dst, N);
    reinit_dst(i);
    if (t[0] != t[1])
      abort();
  }

  // Test correctness of memset
  for (int i = 0; i < 67; i++) {
    int t[2];
    if (!__arm_sc_memset(dst, src[i], i))
      abort();
    t[0] = sum(dst, N);
    reinit_dst(i);
    memset(dst, src[i], i);
    t[1] = sum(dst, N);
    reinit_dst(i);
    if (t[0] != t[1])
      abort();
  }

  // Test correctness of memchr
  for (int i = 0; i < 67; i++) {
    for (int j = 0; j < 67; j++) {
      uint8_t *t[2];
      t[0] = __arm_sc_memchr(src, src[j], i);
      t[1] = memchr(src, src[j], i);
      if (t[0] != t[1])
        abort();
    }
  }

  // Test correctness for memmove
  for (int i = 0; i < 67; i++) {
    for (int j = 0; j < 67; j++) {
      int t[2];
      if (!__arm_sc_memmove(&dst[66 - j], &dst[j], i))
        abort();
      t[0] = sum(dst, N);
      reinit_dst(200);
      memmove(&dst[66 - j], &dst[j], i);
      t[1] = sum(dst, N);
      reinit_dst(200);
      if (t[0] != t[1])
        abort();
    }
  }

  return 0;
}
