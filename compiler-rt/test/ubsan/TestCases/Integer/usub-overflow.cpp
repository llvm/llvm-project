// RUN: %clangxx -DSUB_I32 -fsanitize=unsigned-integer-overflow %s -o %t1 && %run %t1 2>&1 | FileCheck %s --check-prefix=CHECK-SUB_I32
// RUN: %clangxx -DSUB_I64 -fsanitize=unsigned-integer-overflow %s -o %t2 && %run %t2 2>&1 | FileCheck %s --check-prefix=CHECK-SUB_I64
// RUN: %clangxx -DSUB_I128 -fsanitize=unsigned-integer-overflow %s -o %t3 && %run %t3 2>&1 | FileCheck %s --check-prefix=CHECK-SUB_I128

#include <stdint.h>
#include <stdio.h>

int main() {
  // These promote to 'int'.
  (void)(uint8_t(0) - uint8_t(0x7f));
  (void)(uint16_t(0) - uint16_t(0x7fff));

#ifdef SUB_I32
  (void)(uint32_t(1) - uint32_t(2));
  // CHECK-SUB_I32: usub-overflow.cpp:[[@LINE-1]]:22: runtime error: unsigned integer overflow: 1 - 2 cannot be represented in type '{{uint32_t|unsigned int}}'
#endif

#ifdef SUB_I64
  (void)(uint64_t(8000000000000000000ll) - uint64_t(9000000000000000000ll));
  // CHECK-SUB_I64: 8000000000000000000 - 9000000000000000000 cannot be represented in type '{{uint64_t|unsigned long( long)?}}'
#endif

#ifdef SUB_I128
# if defined(__SIZEOF_INT128__) && !defined(_WIN32)
  (void)((__uint128_t(1) << 126) - (__uint128_t(1) << 127));
# else
  puts("__int128 not supported\n");
# endif
  // CHECK-SUB_I128: {{0x40000000000000000000000000000000 - 0x80000000000000000000000000000000 cannot be represented in type '__uint128_t'|__int128 not supported}}
#endif
}
