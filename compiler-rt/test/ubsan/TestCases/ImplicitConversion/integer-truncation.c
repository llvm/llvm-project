// RUN: %clang   -x c   -fsanitize=implicit-integer-truncation %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clangxx -x c++ -fsanitize=implicit-integer-truncation %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK

#include <stdint.h>

#if !defined(__cplusplus)
#define bool _Bool
#endif

int main() {
// CHECK-NOT: integer-truncation.c

  // Negative tests. Even if they produce unexpected results, this sanitizer does not care.
  int8_t n0 = (~((uint32_t)0)); // ~0 -> -1, but do not warn.
  uint8_t n2 = 128;
  uint8_t n3 = 255;
  // Bools do not count
  bool b0 = (~((uint32_t)0));
  bool b1 = 255;

  // Explicit and-ing of bits will silence it.
  uint8_t nc0 = (~((uint32_t)0)) & 255;

  // Explicit casts
  uint8_t i0 = (uint8_t)(~((uint32_t)0));

#if defined(__cplusplus)
  uint8_t i1 = uint8_t(~(uint32_t(0)));
  uint8_t i2 = static_cast<uint8_t>(~(uint32_t(0)));
#endif

  // Positive tests.

  uint8_t t_b0 = (~((uint16_t)(0)));
// CHECK: {{.*}}integer-truncation.c:[[@LINE-1]]:18: runtime error: implicit conversion from type 'int' of value -1 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 255 (8-bit, unsigned)

  uint8_t t_b1 = (~((uint32_t)0));
// CHECK: {{.*}}integer-truncation.c:[[@LINE-1]]:18: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type 'uint8_t' (aka 'unsigned char') changed the value to 255 (8-bit, unsigned)
  uint16_t t_b2 = (~((uint32_t)0));
// CHECK: {{.*}}integer-truncation.c:[[@LINE-1]]:19: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type 'uint16_t' (aka 'unsigned short') changed the value to 65535 (16-bit, unsigned)

  uint8_t t_b3 = ~((uint64_t)0);
// CHECK: {{.*}}integer-truncation.c:[[@LINE-1]]:18: runtime error: implicit conversion from type 'uint64_t' (aka 'unsigned long{{[^']*}}') of value 18446744073709551615 (64-bit, unsigned) to type 'uint8_t' (aka 'unsigned char') changed the value to 255 (8-bit, unsigned)
  uint16_t t_b4 = ~((uint64_t)0);
// CHECK: {{.*}}integer-truncation.c:[[@LINE-1]]:19: runtime error: implicit conversion from type 'uint64_t' (aka 'unsigned long{{[^']*}}') of value 18446744073709551615 (64-bit, unsigned) to type 'uint16_t' (aka 'unsigned short') changed the value to 65535 (16-bit, unsigned)
  uint32_t t_b5 = ~((uint64_t)0);
// CHECK: {{.*}}integer-truncation.c:[[@LINE-1]]:19: runtime error: implicit conversion from type 'uint64_t' (aka 'unsigned long{{[^']*}}') of value 18446744073709551615 (64-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 4294967295 (32-bit, unsigned)

  int8_t t1 = 255;
// CHECK: {{.*}}integer-truncation.c:[[@LINE-1]]:15: runtime error: implicit conversion from type 'int' of value 255 (32-bit, signed) to type 'int8_t' (aka 'signed char') changed the value to -1 (8-bit, signed)
  uint8_t t2 = 256;
// CHECK: {{.*}}integer-truncation.c:[[@LINE-1]]:16: runtime error: implicit conversion from type 'int' of value 256 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (8-bit, unsigned)
  int8_t t3 = 256;
// CHECK: {{.*}}integer-truncation.c:[[@LINE-1]]:15: runtime error: implicit conversion from type 'int' of value 256 (32-bit, signed) to type 'int8_t' (aka 'signed char') changed the value to 0 (8-bit, signed)
  uint8_t t4 = 257;
// CHECK: {{.*}}integer-truncation.c:[[@LINE-1]]:16: runtime error: implicit conversion from type 'int' of value 257 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 1 (8-bit, unsigned)
  int8_t t5 = 257;
// CHECK: {{.*}}integer-truncation.c:[[@LINE-1]]:15: runtime error: implicit conversion from type 'int' of value 257 (32-bit, signed) to type 'int8_t' (aka 'signed char') changed the value to 1 (8-bit, signed)
  int8_t t6 = 128;
// CHECK: {{.*}}integer-truncation.c:[[@LINE-1]]:15: runtime error: implicit conversion from type 'int' of value 128 (32-bit, signed) to type 'int8_t' (aka 'signed char') changed the value to -128 (8-bit, signed)

  return 0;
}
