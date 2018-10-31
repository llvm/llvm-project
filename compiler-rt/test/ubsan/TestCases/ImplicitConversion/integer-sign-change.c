// RUN: %clang   -x c   -fsanitize=implicit-integer-sign-change %s -DV0 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V0
// RUN: %clang   -x c   -fsanitize=implicit-integer-sign-change %s -DV1 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V1
// RUN: %clang   -x c   -fsanitize=implicit-integer-sign-change %s -DV2 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V2
// RUN: %clang   -x c   -fsanitize=implicit-integer-sign-change %s -DV3 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V3
// RUN: %clang   -x c   -fsanitize=implicit-integer-sign-change %s -DV4 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V4
// RUN: %clang   -x c   -fsanitize=implicit-integer-sign-change %s -DV5 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V5
// RUN: %clang   -x c   -fsanitize=implicit-integer-sign-change %s -DV6 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V6

// RUN: %clang   -x c++ -fsanitize=implicit-integer-sign-change %s -DV0 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V0
// RUN: %clang   -x c++ -fsanitize=implicit-integer-sign-change %s -DV1 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V1
// RUN: %clang   -x c++ -fsanitize=implicit-integer-sign-change %s -DV2 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V2
// RUN: %clang   -x c++ -fsanitize=implicit-integer-sign-change %s -DV3 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V3
// RUN: %clang   -x c++ -fsanitize=implicit-integer-sign-change %s -DV4 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V4
// RUN: %clang   -x c++ -fsanitize=implicit-integer-sign-change %s -DV5 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V5
// RUN: %clang   -x c++ -fsanitize=implicit-integer-sign-change %s -DV6 -o %t && %run %t 2>&1 | FileCheck %s --implicit-check-not="implicit conversion" --check-prefixes=CHECK-V6

#include <stdint.h>

// Test plan:
//  * Two types - int and char
//  * Two signs - signed and unsigned
//  * Square that - we have input and output types.
// Thus, there are total of (2*2)^2 == 16 tests.
// These are all the possible variations/combinations of casts.
// However, not all of them should result in the check.
// So here, we *only* check which should and which should not result in checks.

//============================================================================//
// Half of the cases do not need the check.                                   //
//============================================================================//

uint32_t negative0_convert_unsigned_int_to_unsigned_int(uint32_t x) {
  return x;
}

uint8_t negative1_convert_unsigned_char_to_unsigned_char(uint8_t x) {
  return x;
}

int32_t negative2_convert_signed_int_to_signed_int(int32_t x) {
  return x;
}

int8_t negative3_convert_signed_char_to_signed_char(int8_t x) {
  return x;
}

uint8_t negative4_convert_unsigned_int_to_unsigned_char(uint32_t x) {
  return x;
}

uint32_t negative5_convert_unsigned_char_to_unsigned_int(uint8_t x) {
  return x;
}

int32_t negative6_convert_unsigned_char_to_signed_int(uint8_t x) {
  return x;
}

int32_t negative7_convert_signed_char_to_signed_int(int8_t x) {
  return x;
}

void test_negatives() {
  // No bits set.
  negative0_convert_unsigned_int_to_unsigned_int(0);
  negative1_convert_unsigned_char_to_unsigned_char(0);
  negative2_convert_signed_int_to_signed_int(0);
  negative3_convert_signed_char_to_signed_char(0);
  negative4_convert_unsigned_int_to_unsigned_char(0);
  negative5_convert_unsigned_char_to_unsigned_int(0);
  negative6_convert_unsigned_char_to_signed_int(0);
  negative7_convert_signed_char_to_signed_int(0);

  // One lowest bit set.
  negative0_convert_unsigned_int_to_unsigned_int(1);
  negative1_convert_unsigned_char_to_unsigned_char(1);
  negative2_convert_signed_int_to_signed_int(1);
  negative3_convert_signed_char_to_signed_char(1);
  negative4_convert_unsigned_int_to_unsigned_char(1);
  negative5_convert_unsigned_char_to_unsigned_int(1);
  negative6_convert_unsigned_char_to_signed_int(1);
  negative7_convert_signed_char_to_signed_int(1);

  // All source bits set.
  negative0_convert_unsigned_int_to_unsigned_int((uint32_t)UINT32_MAX);
  negative1_convert_unsigned_char_to_unsigned_char((uint8_t)UINT8_MAX);
  negative2_convert_signed_int_to_signed_int((int32_t)INT32_MAX);
  negative3_convert_signed_char_to_signed_char((int8_t)INT8_MAX);
  negative4_convert_unsigned_int_to_unsigned_char((uint32_t)UINT32_MAX);
  negative5_convert_unsigned_char_to_unsigned_int((uint8_t)UINT8_MAX);
  negative6_convert_unsigned_char_to_signed_int((uint8_t)UINT8_MAX);
  negative7_convert_signed_char_to_signed_int((int8_t)INT8_MAX);

  // Source 'sign' bit set.
  negative0_convert_unsigned_int_to_unsigned_int((uint32_t)INT32_MIN);
  negative1_convert_unsigned_char_to_unsigned_char((uint8_t)INT8_MIN);
  negative2_convert_signed_int_to_signed_int((int32_t)INT32_MIN);
  negative3_convert_signed_char_to_signed_char((int8_t)INT8_MIN);
  negative4_convert_unsigned_int_to_unsigned_char((uint32_t)INT32_MIN);
  negative5_convert_unsigned_char_to_unsigned_int((uint8_t)INT8_MIN);
  negative6_convert_unsigned_char_to_signed_int((uint8_t)INT8_MIN);
  negative7_convert_signed_char_to_signed_int((int8_t)INT8_MIN);
}

//============================================================================//
// The remaining 8 cases *do* need the check.   //
//============================================================================//

int32_t positive0_convert_unsigned_int_to_signed_int(uint32_t x) {
#line 100
  return x;
}

uint32_t positive1_convert_signed_int_to_unsigned_int(int32_t x) {
#line 200
  return x;
}

uint8_t positive2_convert_signed_int_to_unsigned_char(int32_t x) {
#line 300
  return x;
}

uint8_t positive3_convert_signed_char_to_unsigned_char(int8_t x) {
#line 400
  return x;
}

int8_t positive4_convert_unsigned_char_to_signed_char(uint8_t x) {
#line 500
  return x;
}

uint32_t positive5_convert_signed_char_to_unsigned_int(int8_t x) {
#line 600
  return x;
}

int8_t positive6_convert_unsigned_int_to_signed_char(uint32_t x) {
#line 700
  return x;
}

int8_t positive7_convert_signed_int_to_signed_char(int32_t x) {
#line 800
  return x;
}

#line 1120 // !!!

void test_positives() {
  // No bits set.
  positive0_convert_unsigned_int_to_signed_int(0);
  positive1_convert_signed_int_to_unsigned_int(0);
  positive2_convert_signed_int_to_unsigned_char(0);
  positive3_convert_signed_char_to_unsigned_char(0);
  positive4_convert_unsigned_char_to_signed_char(0);
  positive5_convert_signed_char_to_unsigned_int(0);
  positive6_convert_unsigned_int_to_signed_char(0);
  positive7_convert_signed_int_to_signed_char(0);

  // One lowest bit set.
  positive0_convert_unsigned_int_to_signed_int(1);
  positive1_convert_signed_int_to_unsigned_int(1);
  positive2_convert_signed_int_to_unsigned_char(1);
  positive3_convert_signed_char_to_unsigned_char(1);
  positive4_convert_unsigned_char_to_signed_char(1);
  positive5_convert_signed_char_to_unsigned_int(1);
  positive6_convert_unsigned_int_to_signed_char(1);
  positive7_convert_signed_int_to_signed_char(1);

#if defined(V0)
  // All source bits set.
  positive0_convert_unsigned_int_to_signed_int((uint32_t)UINT32_MAX);
// CHECK-V0: {{.*}}integer-sign-change.c:100:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type 'int32_t' (aka 'int') changed the value to -1 (32-bit, signed)
  positive1_convert_signed_int_to_unsigned_int((int32_t)UINT32_MAX);
// CHECK-V0: {{.*}}integer-sign-change.c:200:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -1 (32-bit, signed) to type 'uint32_t' (aka 'unsigned int') changed the value to 4294967295 (32-bit, unsigned)
  positive2_convert_signed_int_to_unsigned_char((int32_t)UINT32_MAX);
// CHECK-V0: {{.*}}integer-sign-change.c:300:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -1 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 255 (8-bit, unsigned)
  positive3_convert_signed_char_to_unsigned_char((int8_t)UINT8_MAX);
// CHECK-V0: {{.*}}integer-sign-change.c:400:10: runtime error: implicit conversion from type 'int8_t' (aka 'signed char') of value -1 (8-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 255 (8-bit, unsigned)
  positive4_convert_unsigned_char_to_signed_char((uint8_t)UINT8_MAX);
// CHECK-V0: {{.*}}integer-sign-change.c:500:10: runtime error: implicit conversion from type 'uint8_t' (aka 'unsigned char') of value 255 (8-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -1 (8-bit, signed)
  positive5_convert_signed_char_to_unsigned_int((int8_t)UINT8_MAX);
// CHECK-V0: {{.*}}integer-sign-change.c:600:10: runtime error: implicit conversion from type 'int8_t' (aka 'signed char') of value -1 (8-bit, signed) to type 'uint32_t' (aka 'unsigned int') changed the value to 4294967295 (32-bit, unsigned)
  positive6_convert_unsigned_int_to_signed_char((uint32_t)UINT32_MAX);
// CHECK-V0: {{.*}}integer-sign-change.c:700:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -1 (8-bit, signed)
  positive7_convert_signed_int_to_signed_char((int32_t)UINT32_MAX);
#elif defined(V1)
  // Source 'Sign' bit set.
  positive0_convert_unsigned_int_to_signed_int((uint32_t)INT32_MIN);
// CHECK-V1: {{.*}}integer-sign-change.c:100:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 2147483648 (32-bit, unsigned) to type 'int32_t' (aka 'int') changed the value to -2147483648 (32-bit, signed)
  positive1_convert_signed_int_to_unsigned_int((int32_t)INT32_MIN);
// CHECK-V1: {{.*}}integer-sign-change.c:200:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -2147483648 (32-bit, signed) to type 'uint32_t' (aka 'unsigned int') changed the value to 2147483648 (32-bit, unsigned)
  positive2_convert_signed_int_to_unsigned_char((int32_t)INT32_MIN);
// CHECK-V1: {{.*}}integer-sign-change.c:300:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -2147483648 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (8-bit, unsigned)
  positive3_convert_signed_char_to_unsigned_char((int8_t)INT8_MIN);
// CHECK-V1: {{.*}}integer-sign-change.c:400:10: runtime error: implicit conversion from type 'int8_t' (aka 'signed char') of value -128 (8-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 128 (8-bit, unsigned)
  positive4_convert_unsigned_char_to_signed_char((uint8_t)INT8_MIN);
// CHECK-V1: {{.*}}integer-sign-change.c:500:10: runtime error: implicit conversion from type 'uint8_t' (aka 'unsigned char') of value 128 (8-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -128 (8-bit, signed)
  positive5_convert_signed_char_to_unsigned_int((int8_t)INT8_MIN);
// CHECK-V1: {{.*}}integer-sign-change.c:600:10: runtime error: implicit conversion from type 'int8_t' (aka 'signed char') of value -128 (8-bit, signed) to type 'uint32_t' (aka 'unsigned int') changed the value to 4294967168 (32-bit, unsigned)
  positive6_convert_unsigned_int_to_signed_char((uint32_t)INT32_MIN);
  positive7_convert_signed_int_to_signed_char((int32_t)INT32_MIN);
// CHECK-V1: {{.*}}integer-sign-change.c:800:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -2147483648 (32-bit, signed) to type 'int8_t' (aka 'signed char') changed the value to 0 (8-bit, signed)
#elif defined(V2)
  // All bits except the source 'Sign' bit are set.
  positive0_convert_unsigned_int_to_signed_int((uint32_t)INT32_MAX);
  positive1_convert_signed_int_to_unsigned_int((int32_t)INT32_MAX);
  positive2_convert_signed_int_to_unsigned_char((int32_t)INT32_MAX);
  positive3_convert_signed_char_to_unsigned_char((int8_t)INT8_MAX);
  positive4_convert_unsigned_char_to_signed_char((uint8_t)INT8_MAX);
  positive5_convert_signed_char_to_unsigned_int((int8_t)INT8_MAX);
  positive6_convert_unsigned_int_to_signed_char((uint32_t)INT32_MAX);
// CHECK-V2: {{.*}}integer-sign-change.c:700:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 2147483647 (32-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -1 (8-bit, signed)
  positive7_convert_signed_int_to_signed_char((int32_t)INT32_MAX);
// CHECK-V2: {{.*}}integer-sign-change.c:800:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value 2147483647 (32-bit, signed) to type 'int8_t' (aka 'signed char') changed the value to -1 (8-bit, signed)
#elif defined(V3)
  // All destination bits set.
  positive0_convert_unsigned_int_to_signed_int((uint32_t)UINT32_MAX);
// CHECK-V3: {{.*}}integer-sign-change.c:100:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type 'int32_t' (aka 'int') changed the value to -1 (32-bit, signed)
  positive1_convert_signed_int_to_unsigned_int((int32_t)UINT32_MAX);
// CHECK-V3: {{.*}}integer-sign-change.c:200:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -1 (32-bit, signed) to type 'uint32_t' (aka 'unsigned int') changed the value to 4294967295 (32-bit, unsigned)
  positive2_convert_signed_int_to_unsigned_char((int32_t)UINT8_MAX);
  positive3_convert_signed_char_to_unsigned_char((int8_t)UINT8_MAX);
// CHECK-V3: {{.*}}integer-sign-change.c:400:10: runtime error: implicit conversion from type 'int8_t' (aka 'signed char') of value -1 (8-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 255 (8-bit, unsigned)
  positive4_convert_unsigned_char_to_signed_char((uint8_t)UINT8_MAX);
// CHECK-V3: {{.*}}integer-sign-change.c:500:10: runtime error: implicit conversion from type 'uint8_t' (aka 'unsigned char') of value 255 (8-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -1 (8-bit, signed)
  positive5_convert_signed_char_to_unsigned_int((int8_t)UINT32_MAX);
// CHECK-V3: {{.*}}integer-sign-change.c:600:10: runtime error: implicit conversion from type 'int8_t' (aka 'signed char') of value -1 (8-bit, signed) to type 'uint32_t' (aka 'unsigned int') changed the value to 4294967295 (32-bit, unsigned)
  positive6_convert_unsigned_int_to_signed_char((uint32_t)UINT8_MAX);
// CHECK-V3: {{.*}}integer-sign-change.c:700:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 255 (32-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -1 (8-bit, signed)
  positive7_convert_signed_int_to_signed_char((int32_t)UINT8_MAX);
// CHECK-V3: {{.*}}integer-sign-change.c:800:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value 255 (32-bit, signed) to type 'int8_t' (aka 'signed char') changed the value to -1 (8-bit, signed)
#elif defined(V4)
  // Destination 'sign' bit set.
  positive0_convert_unsigned_int_to_signed_int((uint32_t)INT32_MIN);
// CHECK-V4: {{.*}}integer-sign-change.c:100:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 2147483648 (32-bit, unsigned) to type 'int32_t' (aka 'int') changed the value to -2147483648 (32-bit, signed)
  positive1_convert_signed_int_to_unsigned_int((int32_t)INT32_MIN);
// CHECK-V4: {{.*}}integer-sign-change.c:200:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -2147483648 (32-bit, signed) to type 'uint32_t' (aka 'unsigned int') changed the value to 2147483648 (32-bit, unsigned)
  positive2_convert_signed_int_to_unsigned_char((int32_t)INT8_MIN);
// CHECK-V4: {{.*}}integer-sign-change.c:300:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -128 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 128 (8-bit, unsigned)
  positive3_convert_signed_char_to_unsigned_char((int8_t)INT8_MIN);
// CHECK-V4: {{.*}}integer-sign-change.c:400:10: runtime error: implicit conversion from type 'int8_t' (aka 'signed char') of value -128 (8-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 128 (8-bit, unsigned)
  positive4_convert_unsigned_char_to_signed_char((uint8_t)INT8_MIN);
// CHECK-V4: {{.*}}integer-sign-change.c:500:10: runtime error: implicit conversion from type 'uint8_t' (aka 'unsigned char') of value 128 (8-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -128 (8-bit, signed)
  positive5_convert_signed_char_to_unsigned_int((int8_t)INT32_MIN);
  positive6_convert_unsigned_int_to_signed_char((uint32_t)INT8_MIN);
// CHECK-V4: {{.*}}integer-sign-change.c:700:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967168 (32-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -128 (8-bit, signed)
  positive7_convert_signed_int_to_signed_char((int32_t)INT8_MIN);
#elif defined(V5)
  // All bits except the destination 'sign' bit are set.
  positive0_convert_unsigned_int_to_signed_int((uint32_t)INT32_MIN);
// CHECK-V5: {{.*}}integer-sign-change.c:100:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 2147483648 (32-bit, unsigned) to type 'int32_t' (aka 'int') changed the value to -2147483648 (32-bit, signed)
  positive1_convert_signed_int_to_unsigned_int((int32_t)INT32_MIN);
// CHECK-V5: {{.*}}integer-sign-change.c:200:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -2147483648 (32-bit, signed) to type 'uint32_t' (aka 'unsigned int') changed the value to 2147483648 (32-bit, unsigned)
  positive2_convert_signed_int_to_unsigned_char((int32_t)(~((uint32_t)(uint8_t)INT8_MIN)));
// CHECK-V5: {{.*}}integer-sign-change.c:300:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -129 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 127 (8-bit, unsigned)
  positive3_convert_signed_char_to_unsigned_char((int8_t)(INT8_MIN));
// CHECK-V5: {{.*}}integer-sign-change.c:400:10: runtime error: implicit conversion from type 'int8_t' (aka 'signed char') of value -128 (8-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 128 (8-bit, unsigned)
  positive4_convert_unsigned_char_to_signed_char((uint8_t)(INT8_MIN));
// CHECK-V5: {{.*}}integer-sign-change.c:500:10: runtime error: implicit conversion from type 'uint8_t' (aka 'unsigned char') of value 128 (8-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -128 (8-bit, signed)
  positive5_convert_signed_char_to_unsigned_int((int8_t)(INT32_MIN));
  positive6_convert_unsigned_int_to_signed_char(~((uint32_t)(uint8_t)INT8_MIN));
  positive7_convert_signed_int_to_signed_char((int32_t)(~((uint32_t)((uint8_t)INT8_MIN))));
// CHECK-V5: {{.*}}integer-sign-change.c:800:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -129 (32-bit, signed) to type 'int8_t' (aka 'signed char') changed the value to 127 (8-bit, signed)
#elif defined(V6)
  // Only the source and destination sign bits are set.
  positive0_convert_unsigned_int_to_signed_int((uint32_t)INT32_MIN);
// CHECK-V6: {{.*}}integer-sign-change.c:100:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 2147483648 (32-bit, unsigned) to type 'int32_t' (aka 'int') changed the value to -2147483648 (32-bit, signed)
  positive1_convert_signed_int_to_unsigned_int((int32_t)INT32_MIN);
// CHECK-V6: {{.*}}integer-sign-change.c:200:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -2147483648 (32-bit, signed) to type 'uint32_t' (aka 'unsigned int') changed the value to 2147483648 (32-bit, unsigned)
  positive2_convert_signed_int_to_unsigned_char((int32_t)((uint32_t)INT32_MIN | (uint32_t)((uint8_t)INT8_MIN)));
// CHECK-V6: {{.*}}integer-sign-change.c:300:10: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -2147483520 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 128 (8-bit, unsigned)
  positive3_convert_signed_char_to_unsigned_char((int8_t)INT8_MIN);
// CHECK-V6: {{.*}}integer-sign-change.c:400:10: runtime error: implicit conversion from type 'int8_t' (aka 'signed char') of value -128 (8-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 128 (8-bit, unsigned)
  positive4_convert_unsigned_char_to_signed_char((uint8_t)INT8_MIN);
// CHECK-V6: {{.*}}integer-sign-change.c:500:10: runtime error: implicit conversion from type 'uint8_t' (aka 'unsigned char') of value 128 (8-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -128 (8-bit, signed)
  positive5_convert_signed_char_to_unsigned_int((int8_t)INT8_MIN);
// CHECK-V6: {{.*}}integer-sign-change.c:600:10: runtime error: implicit conversion from type 'int8_t' (aka 'signed char') of value -128 (8-bit, signed) to type 'uint32_t' (aka 'unsigned int') changed the value to 4294967168 (32-bit, unsigned)
  positive6_convert_unsigned_int_to_signed_char((uint32_t)((uint32_t)INT32_MIN | (uint32_t)((uint8_t)INT8_MIN)));
// CHECK-V6: {{.*}}integer-sign-change.c:700:10: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 2147483776 (32-bit, unsigned) to type 'int8_t' (aka 'signed char') changed the value to -128 (8-bit, signed)
  positive7_convert_signed_int_to_signed_char((int32_t)((uint32_t)INT32_MIN | (uint32_t)((uint8_t)INT8_MIN)));
#else
#error Some V* needs to be defined!
#endif
}

// CHECK-NOT: implicit conversion

int main() {
  test_negatives();
  test_positives();

  return 0;
}
