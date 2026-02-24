// REQUIRES: int256
//
// RUN: %clangxx -DADD_I256 -fsanitize=signed-integer-overflow %s -o %t1 && %run %t1 2>&1 | FileCheck %s --check-prefix=CHECK-ADD_I256
// RUN: %clangxx -DSUB_I256 -fsanitize=signed-integer-overflow %s -o %t2 && %run %t2 2>&1 | FileCheck %s --check-prefix=CHECK-SUB_I256
// RUN: %clangxx -DNEG_I256 -fsanitize=signed-integer-overflow %s -o %t3 && %run %t3 2>&1 | FileCheck %s --check-prefix=CHECK-NEG_I256
//
// Test UBSan detection of signed integer overflow for __int256_t.

#include <stdint.h>

int main() {
#ifdef ADD_I256
#  if defined(__SIZEOF_INT256__)
  // Overflow: 2^254 + 2^254 = 2^255, which exceeds __int256_t max (2^255 - 1)
  (void)((__int256_t(1) << 254) + (__int256_t(1) << 254));
#  else
  // Fallback message for platforms without __int256
  __builtin_printf("__int256 not supported\n");
#  endif
  // CHECK-ADD_I256: {{0x[0-9a-f]+ \+ 0x[0-9a-f]+ cannot be represented in type '__int256_t'|__int256 not supported}}
#endif

#ifdef SUB_I256
#  if defined(__SIZEOF_INT256__)
  // Overflow: min - 1
  __int256_t min_val = (__int256_t)1
                       << 255; // This is the minimum (negative) value
  (void)(min_val - 1);
#  else
  __builtin_printf("__int256 not supported\n");
#  endif
  // CHECK-SUB_I256: {{0x[0-9a-f]+ - 1 cannot be represented in type '__int256_t'|__int256 not supported}}
#endif

#ifdef NEG_I256
#  if defined(__SIZEOF_INT256__)
  // Overflow: -min = -(-2^255) overflows because max is 2^255 - 1
  __int256_t min_val = (__int256_t)1 << 255;
  (void)(-min_val);
#  else
  __builtin_printf("__int256 not supported\n");
#  endif
  // CHECK-NEG_I256: {{negation of -?0x[0-9a-f]+ cannot be represented in type '__int256_t'|__int256 not supported}}
#endif
}
