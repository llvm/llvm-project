// RUN: %clang   -x c   -fsanitize=implicit-integer-truncation %s -DV0 -o %t && %run %t 2>&1 | not FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V0
// RUN: %clang   -x c   -fsanitize=implicit-integer-truncation %s -DV1 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V1
// RUN: %clang   -x c   -fsanitize=implicit-integer-truncation %s -DV2 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V2
// RUN: %clang   -x c   -fsanitize=implicit-integer-truncation %s -DV3 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V3
// RUN: %clang   -x c   -fsanitize=implicit-integer-truncation %s -DV4 -o %t && %run %t 2>&1 | not FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V4
// RUN: %clang   -x c   -fsanitize=implicit-integer-truncation %s -DV5 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V5
// RUN: %clang   -x c   -fsanitize=implicit-integer-truncation %s -DV6 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V6

// RUN: %clang   -x c++ -fsanitize=implicit-integer-truncation %s -DV0 -o %t && %run %t 2>&1 | not FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V0
// RUN: %clang   -x c++ -fsanitize=implicit-integer-truncation %s -DV1 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V1
// RUN: %clang   -x c++ -fsanitize=implicit-integer-truncation %s -DV2 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V2
// RUN: %clang   -x c++ -fsanitize=implicit-integer-truncation %s -DV3 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V3
// RUN: %clang   -x c++ -fsanitize=implicit-integer-truncation %s -DV4 -o %t && %run %t 2>&1 | not FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V4
// RUN: %clang   -x c++ -fsanitize=implicit-integer-truncation %s -DV5 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V5
// RUN: %clang   -x c++ -fsanitize=implicit-integer-truncation %s -DV6 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V6

#include <stdint.h>

int8_t positive6_convert_unsigned_int_to_signed_char(uint32_t x) {
#line 100
  return x;
}

#line 1120 // !!!

void test_positives() {
  // No bits set.
  positive6_convert_unsigned_int_to_signed_char(0);

  // One lowest bit set.
  positive6_convert_unsigned_int_to_signed_char(1);

#if defined(V0)
  // All source bits set.
  positive6_convert_unsigned_int_to_signed_char((uint32_t)UINT32_MAX);
#elif defined(V1)
  // Source 'Sign' bit set.
  positive6_convert_unsigned_int_to_signed_char((uint32_t)INT32_MIN);
// CHECK-V1: {{.*}}integer-conversion.c:100:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 2147483648 (32-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to 0 (8-bit, signed)
#elif defined(V2)
  // All bits except the source 'Sign' bit are set.
  positive6_convert_unsigned_int_to_signed_char((uint32_t)INT32_MAX);
// CHECK-V2: {{.*}}integer-conversion.c:100:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 2147483647 (32-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -1 (8-bit, signed)
#elif defined(V3)
  // All destination bits set.
  positive6_convert_unsigned_int_to_signed_char((uint32_t)UINT8_MAX);
// CHECK-V3: {{.*}}integer-conversion.c:100:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 255 (32-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -1 (8-bit, signed)
#elif defined(V4)
  // Destination 'sign' bit set.
  positive6_convert_unsigned_int_to_signed_char((uint32_t)INT8_MIN);
#elif defined(V5)
  // All bits except the destination 'sign' bit are set.
  positive6_convert_unsigned_int_to_signed_char(~((uint32_t)(uint8_t)INT8_MIN));
// CHECK-V5: {{.*}}integer-conversion.c:100:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967167 (32-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to 127 (8-bit, signed)
#elif defined(V6)
  // Only the source and destination sign bits are set.
  positive6_convert_unsigned_int_to_signed_char((uint32_t)((uint32_t)INT32_MIN | (uint32_t)((uint8_t)INT8_MIN)));
// CHECK-V6: {{.*}}integer-conversion.c:100:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 2147483776 (32-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -128 (8-bit, signed)
#else
#error Some V* needs to be defined!
#endif
}

// CHECK-NOT: implicit conversion

int main() {
  test_positives();

  return 0;
}
