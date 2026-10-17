// REQUIRES: librt_has_sync_fetch_and_max_4, librt_has_sync_fetch_and_max_8
// RUN: %clang_builtins %s %librt -o %t && %run %t

#include <stdint.h>
#include <stdio.h>

// Call the library functions through asm labels, as calls to the
// __sync_fetch_and_* builtins would be expanded inline instead.
#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
#define EXTERNAL_NAME(name) __asm__(STRINGIFY(__USER_LABEL_PREFIX__) #name)

#define DECLARE(op, size, type)                                                \
  type sync_fetch_and_##op##_##size(type *ptr, type val)                       \
      EXTERNAL_NAME(__sync_fetch_and_##op##_##size);

DECLARE(max, 4, int32_t)
DECLARE(min, 4, int32_t)
DECLARE(umax, 4, uint32_t)
DECLARE(umin, 4, uint32_t)
#if __ARM_ARCH_PROFILE != 'M'
DECLARE(max, 8, int64_t)
DECLARE(min, 8, int64_t)
DECLARE(umax, 8, uint64_t)
DECLARE(umin, 8, uint64_t)
#endif

// Include values that differ only in the low word, only in the high word, and
// only in the sign.
static const uint64_t values[] = {
    0x0000000000000000ULL, 0x0000000000000003ULL, 0x0000000000000005ULL,
    0x000000007FFFFFFFULL, 0x0000000080000000ULL, 0x00000000FFFFFFFFULL,
    0x0000000100000000ULL, 0x0000000100000003ULL, 0x7FFFFFFFFFFFFFFFULL,
    0x8000000000000000ULL, 0xFFFFFFFF00000005ULL, 0xFFFFFFFFFFFFFFFFULL,
};

#define N (sizeof(values) / sizeof(values[0]))

#define TEST(op, size, type, expected)                                         \
  do {                                                                         \
    type a = (type)values[i], b = (type)values[j];                             \
    type x = a;                                                                \
    type r = sync_fetch_and_##op##_##size(&x, b);                              \
    if (r != a || x != (expected)) {                                           \
      printf("error in __sync_fetch_and_" #op "_" #size "(0x%llX, 0x%llX): "   \
             "returned 0x%llX, stored 0x%llX\n",                               \
             (unsigned long long)a, (unsigned long long)b,                     \
             (unsigned long long)r, (unsigned long long)x);                    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(void) {
  for (unsigned i = 0; i < N; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      TEST(max, 4, int32_t, a > b ? a : b);
      TEST(min, 4, int32_t, a < b ? a : b);
      TEST(umax, 4, uint32_t, a > b ? a : b);
      TEST(umin, 4, uint32_t, a < b ? a : b);
#if __ARM_ARCH_PROFILE != 'M'
      TEST(max, 8, int64_t, a > b ? a : b);
      TEST(min, 8, int64_t, a < b ? a : b);
      TEST(umax, 8, uint64_t, a > b ? a : b);
      TEST(umin, 8, uint64_t, a < b ? a : b);
#endif
    }
  }
  return 0;
}
