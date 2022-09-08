// RUN: %clangxx -DADD_I32 -fsanitize=unsigned-integer-overflow %s -o %t1 && %run %t1 2>&1 | FileCheck %s --check-prefix=CHECK-ADD_I32
// RUN: %clangxx -DADD_I64 -fsanitize=unsigned-integer-overflow %s -o %t2 && %run %t2 2>&1 | FileCheck %s --check-prefix=CHECK-ADD_I64
// RUN: %clangxx -DADD_I128 -fsanitize=unsigned-integer-overflow %s -o %t3 && %run %t3 2>&1 | FileCheck %s --check-prefix=CHECK-ADD_I128

#include <stdint.h>
#include <stdio.h>

int main() {
  // These promote to 'int'.
  (void)(uint8_t(0xff) + uint8_t(0xff));
  (void)(uint16_t(0xf0fff) + uint16_t(0x0fff));

#ifdef ADD_I32
  uint32_t k = 0x87654321;
  k += 0xedcba987;
  // CHECK-ADD_I32: uadd-overflow.cpp:[[@LINE-1]]:5: runtime error: unsigned integer overflow: 2271560481 + 3989547399 cannot be represented in type 'unsigned int'
#endif

#ifdef ADD_I64
  (void)(uint64_t(10000000000000000000ull) + uint64_t(9000000000000000000ull));
  // CHECK-ADD_I64: 10000000000000000000 + 9000000000000000000 cannot be represented in type '{{uint64_t|unsigned long( long)?}}'
#endif

#ifdef ADD_I128
# if defined(__SIZEOF_INT128__) && !defined(_WIN32)
  (void)((__uint128_t(1) << 127) + (__uint128_t(1) << 127));
# else
  puts("__int128 not supported");
# endif
  // CHECK-ADD_I128: {{0x80000000000000000000000000000000 \+ 0x80000000000000000000000000000000 cannot be represented in type '__uint128_t'|__int128 not supported}}
#endif
}
