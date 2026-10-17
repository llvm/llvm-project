// REQUIRES: librt_has_sync_fetch_and_add_4, librt_has_sync_fetch_and_add_8
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

DECLARE(add, 4, uint32_t)
DECLARE(sub, 4, uint32_t)
#if __ARM_ARCH_PROFILE != 'M'
DECLARE(add, 8, uint64_t)
DECLARE(sub, 8, uint64_t)
#endif

// Include values whose sums and differences carry or borrow across 32 bits.
static const uint64_t values[] = {
    0x0000000000000000ULL, 0x0000000000000001ULL, 0x00000000FFFFFFFFULL,
    0x0000000100000000ULL, 0x0123456789ABCDEFULL, 0x7FFFFFFFFFFFFFFFULL,
    0x8000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL,
};

#define N (sizeof(values) / sizeof(values[0]))

#define TEST(op, size, type, expected)                                         \
  do {                                                                         \
    type a = (type)values[i], b = (type)values[j];                             \
    type x = a;                                                                \
    type r = sync_fetch_and_##op##_##size(&x, b);                              \
    if (r != a || x != (type)(expected)) {                                     \
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
      TEST(add, 4, uint32_t, a + b);
      TEST(sub, 4, uint32_t, a - b);
#if __ARM_ARCH_PROFILE != 'M'
      TEST(add, 8, uint64_t, a + b);
      TEST(sub, 8, uint64_t, a - b);
#endif
    }
  }
  return 0;
}
