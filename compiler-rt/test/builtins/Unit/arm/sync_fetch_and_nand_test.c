// REQUIRES: librt_has_sync_fetch_and_nand_4, librt_has_sync_fetch_and_nand_8
// RUN: %clang_builtins %s %librt -o %t && %run %t

#include <stdint.h>
#include <stdio.h>

// Call the library functions through asm labels, as calls to the
// __sync_fetch_and_nand builtins would be expanded inline instead.
#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)
#define EXTERNAL_NAME(name) __asm__(STRINGIFY(__USER_LABEL_PREFIX__) #name)

uint32_t sync_fetch_and_nand_4(uint32_t *ptr, uint32_t val)
    EXTERNAL_NAME(__sync_fetch_and_nand_4);
#if __ARM_ARCH_PROFILE != 'M'
uint64_t sync_fetch_and_nand_8(uint64_t *ptr, uint64_t val)
    EXTERNAL_NAME(__sync_fetch_and_nand_8);
#endif

static const uint64_t values[] = {
    0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0x0123456789ABCDEFULL,
    0xFEDCBA9876543210ULL, 0x00000000FFFFFFFFULL, 0x8000000000000001ULL,
};

#define N (sizeof(values) / sizeof(values[0]))

int main(void) {
  for (unsigned i = 0; i < N; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      // __sync_fetch_and_nand stores ~(*ptr & val) and returns the old value.
      uint32_t a32 = (uint32_t)values[i], b32 = (uint32_t)values[j];
      uint32_t x32 = a32;
      uint32_t r32 = sync_fetch_and_nand_4(&x32, b32);
      if (r32 != a32 || x32 != (uint32_t)~(a32 & b32)) {
        printf("error in __sync_fetch_and_nand_4(0x%08X, 0x%08X): "
               "returned 0x%08X, stored 0x%08X\n",
               a32, b32, r32, x32);
        return 1;
      }
#if __ARM_ARCH_PROFILE != 'M'
      uint64_t a64 = values[i], b64 = values[j];
      uint64_t x64 = a64;
      uint64_t r64 = sync_fetch_and_nand_8(&x64, b64);
      if (r64 != a64 || x64 != ~(a64 & b64)) {
        printf("error in __sync_fetch_and_nand_8(0x%016llX, 0x%016llX): "
               "returned 0x%016llX, stored 0x%016llX\n",
               (unsigned long long)a64, (unsigned long long)b64,
               (unsigned long long)r64, (unsigned long long)x64);
        return 1;
      }
#endif
    }
  }
  return 0;
}
