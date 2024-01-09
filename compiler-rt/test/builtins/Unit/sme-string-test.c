// REQUIRES: linux, aarch64-target-arch
// RUN: %clang_builtins %s %librt -o %t && %run %t

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1024
#define NREPS 1234

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

long get_time_diff(struct timespec tv[2]) {
  long us0 = (tv[0].tv_sec * 1000000) + (tv[0].tv_nsec / 1000);
  long us1 = (tv[1].tv_sec * 1000000) + (tv[1].tv_nsec / 1000);
  return us1 - us0;
}

int main() {
  struct timespec tv[2];

  init();

  // Test correctness of memcpy
  for (int i = 0; i < 67; i++) {
    int t[2];
    if (!__arm_sc_memcpy(dst, src, i)) {
      fprintf(stderr, "Unexpected NULL pointer from __arm_sc_memcpy!\n");
      abort();
    }
    t[0] = sum(dst, N);
    reinit_dst(i);
    memcpy(dst, src, i);
    t[1] = sum(dst, N);
    reinit_dst(i);
    if (t[0] != t[1]) {
      fprintf(stderr, "__arm_sc_memcpy doesn't match memcpy behaviour!\n");
      abort();
    }
  }

#ifdef TEST_PERF
  // Collect perf data for memcpy
  clock_gettime(CLOCK_REALTIME, &tv[0]);
  for (int r = 0; r < NREPS; r++) {
    for (int i = 0; i < 67; i++) {
      int t[2];
      if (!__arm_sc_memcpy(dst, src, i)) {
        fprintf(stderr, "Unexpected NULL pointer from __arm_sc_memcpy!\n");
        abort();
      }
    }
  }
  reinit_dst(67);
  clock_gettime(CLOCK_REALTIME, &tv[1]);
  printf("memcpy time = %ld\n", get_time_diff(tv));
#endif

  // Test correctness of memset
  for (int i = 0; i < 67; i++) {
    int t[2];
    if (!__arm_sc_memset(dst, src[i], i)) {
      fprintf(stderr, "Unexpected NULL pointer from __arm_sc_memset!\n");
      abort();
    }
    t[0] = sum(dst, N);
    reinit_dst(i);
    memset(dst, src[i], i);
    t[1] = sum(dst, N);
    reinit_dst(i);
    if (t[0] != t[1]) {
      fprintf(stderr, "__arm_sc_memcpy doesn't match memset behaviour!\n");
      abort();
    }
  }

#ifdef TEST_PERF
  // Collect perf data for memset
  clock_gettime(CLOCK_REALTIME, &tv[0]);
  for (int r = 0; r < NREPS; r++) {
    for (int i = 0; i < 67; i++) {
      if (!__arm_sc_memset(dst, src[i], i)) {
        fprintf(stderr, "Unexpected NULL pointer from __arm_sc_memset!\n");
        abort();
      }
    }
  }
  reinit_dst(67);
  clock_gettime(CLOCK_REALTIME, &tv[1]);
  printf("memset time = %ld\n", get_time_diff(tv));
#endif

  // Test correctness of memchr
  for (int i = 0; i < 67; i++) {
    for (int j = 0; j < 67; j++) {
      uint8_t *t[2];
      t[0] = __arm_sc_memchr(src, src[j], i);
      t[1] = memchr(src, src[j], i);
      if (t[0] != t[1]) {
        fprintf(stderr, "__arm_sc_memchr doesn't match memchr behaviour!\n");
        abort();
      }
    }
  }

#ifdef TEST_PERF
  // Collect perf data for memchr
  clock_gettime(CLOCK_REALTIME, &tv[0]);
  for (int r = 0; r < NREPS; r++) {
    for (int i = 0; i < 67; i++) {
      for (int j = 0; j < 67; j++) {
        __arm_sc_memchr(src, src[j], i);
      }
    }
  }
  clock_gettime(CLOCK_REALTIME, &tv[1]);
  printf("memchr time = %ld\n", get_time_diff(tv));
#endif

  // Test correctness for memmove
  for (int i = 0; i < 67; i++) {
    for (int j = 0; j < 67; j++) {
      int t[2];
      if (!__arm_sc_memmove(&dst[66 - j], &dst[j], i)) {
        fprintf(stderr, "Unexpected NULL pointer from __arm_sc_memmove!\n");
        abort();
      }
      t[0] = sum(dst, N);
      reinit_dst(200);
      memmove(&dst[66 - j], &dst[j], i);
      t[1] = sum(dst, N);
      reinit_dst(200);
      if (t[0] != t[1]) {
        fprintf(stderr, "__arm_sc_memmove doesn't match memmove behaviour!\n");
        abort();
      }
    }
  }

#ifdef TEST_PERF
  // Collect perf data for memmove
  clock_gettime(CLOCK_REALTIME, &tv[0]);
  for (int r = 0; r < NREPS; r++) {
    for (int i = 0; i < 67; i++) {
      for (int j = 0; j < 67; j++) {
        __arm_sc_memmove(&dst[66 - j], &dst[j], i);
      }
    }
  }
  clock_gettime(CLOCK_REALTIME, &tv[1]);
  printf("memmove time = %ld\n", get_time_diff(tv));
#endif

  return 0;
}
