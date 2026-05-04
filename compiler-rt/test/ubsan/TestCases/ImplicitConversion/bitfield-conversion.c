// RUN: %clang   -x c   -fsanitize=implicit-bitfield-conversion -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clang   -x c   -fsanitize=implicit-bitfield-conversion -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clang   -x c   -fsanitize=implicit-bitfield-conversion -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clang   -x c   -fsanitize=implicit-bitfield-conversion -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clang   -x c   -fsanitize=implicit-conversion          -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK

// RUN: %clangxx -x c++ -fsanitize=implicit-bitfield-conversion -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clangxx -x c++ -fsanitize=implicit-bitfield-conversion -O1 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clangxx -x c++ -fsanitize=implicit-bitfield-conversion -O2 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clangxx -x c++ -fsanitize=implicit-bitfield-conversion -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK
// RUN: %clangxx -x c++ -fsanitize=implicit-conversion          -O0 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK

#include <stdbool.h>
#include <stdint.h>

#define UINT4_MIN 0
#define UINT4_MAX (1 << 4) - 1
#define UINT5_MIN 0
#define UINT5_MAX (1 << 5) - 1
#define INT7_MIN -(1 << 6)
#define INT7_MAX (1 << 6) - 1

typedef struct _X {
  uint8_t a : 4;
  uint32_t b : 5;
  int8_t c : 7;
  int32_t d : 16;
  uint8_t e : 8;
  uint16_t f : 16;
  uint32_t g : 32;
  int8_t h : 8;
  int16_t i : 16;
  int32_t j : 32;
  uint32_t k : 1;
  int32_t l : 1;
  bool m : 1;
} X;

void test_a() {
  X x;
  uint32_t min = UINT4_MIN;
  uint32_t max = UINT4_MAX;

  uint8_t v8 = max + 1;
  uint16_t v16 = (UINT8_MAX + 1) + (max + 1);
  uint32_t v32 = (UINT8_MAX + 1) + (max + 1);

  // Assignment
  x.a = v8;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint8_t' (aka 'unsigned char') of value 16 (8-bit, unsigned) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (4-bit bitfield, unsigned)
  x.a = v16;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint16_t' (aka 'unsigned short') of value 272 (16-bit, unsigned) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (4-bit bitfield, unsigned)
  x.a = v32;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 272 (32-bit, unsigned) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (4-bit bitfield, unsigned)

  // PrePostIncDec
  x.a = min;
  x.a--;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int' of value -1 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 15 (4-bit bitfield, unsigned)
  x.a = min;
  --x.a;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value -1 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 15 (4-bit bitfield, unsigned)

  x.a = max;
  x.a++;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int' of value 16 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (4-bit bitfield, unsigned)
  x.a = max;
  ++x.a;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value 16 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (4-bit bitfield, unsigned)

  x.a = min + 1;
  x.a++;
  x.a = min + 1;
  ++x.a;

  x.a = min + 1;
  x.a--;
  x.a = min + 1;
  --x.a;

  x.a = max - 1;
  x.a++;
  x.a = max - 1;
  ++x.a;

  x.a = max - 1;
  x.a--;
  x.a = max - 1;
  --x.a;

  // Compound assignment
  x.a = 0;
  x.a += max;
  x.a = 0;
  x.a += (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 16 (32-bit, unsigned) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (4-bit bitfield, unsigned)

  x.a = max;
  x.a -= max;
  x.a = max;
  x.a -= (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type 'uint8_t' (aka 'unsigned char') changed the value to 15 (4-bit bitfield, unsigned)

  x.a = 1;
  x.a *= max;
  x.a = 1;
  x.a *= (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 16 (32-bit, unsigned) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (4-bit bitfield, unsigned)
}

void test_b() {
  X x;
  uint32_t min = UINT5_MIN;
  uint32_t max = UINT5_MAX;

  uint8_t v8 = max + 1;
  uint16_t v16 = max + 1;
  uint32_t v32 = max + 1;

  // Assignment
  x.b = v8;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint8_t' (aka 'unsigned char') of value 32 (8-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 0 (5-bit bitfield, unsigned)
  x.b = v16;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint16_t' (aka 'unsigned short') of value 32 (16-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 0 (5-bit bitfield, unsigned)
  x.b = v32;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 32 (32-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 0 (5-bit bitfield, unsigned)

  // PrePostIncDec
  x.b = min;
  x.b--;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 31 (5-bit bitfield, unsigned)
  x.b = min;
  --x.b;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 31 (5-bit bitfield, unsigned)

  x.b = max;
  x.b++;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 32 (32-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 0 (5-bit bitfield, unsigned)
  x.b = max;
  ++x.b;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 32 (32-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 0 (5-bit bitfield, unsigned)

  x.b = min + 1;
  x.b++;
  x.b = min + 1;
  ++x.b;

  x.b = min + 1;
  x.b--;
  x.b = min + 1;
  --x.b;

  x.b = max - 1;
  x.b++;
  x.b = max - 1;
  ++x.b;

  x.b = max - 1;
  x.b--;
  x.b = max - 1;
  --x.b;

  // Compound assignment
  x.b = 0;
  x.b += max;
  x.b = 0;
  x.b += (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 32 (32-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 0 (5-bit bitfield, unsigned)

  x.b = max;
  x.b -= max;
  x.b = max;
  x.b -= (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 31 (5-bit bitfield, unsigned)

  x.b = 1;
  x.b *= max;
  x.b = 1;
  x.b *= (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 32 (32-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 0 (5-bit bitfield, unsigned)
}

void test_c() {
  X x;
  int32_t min = INT7_MIN;
  int32_t max = INT7_MAX;

  uint8_t v8 = max + 1;
  uint16_t v16 = (UINT8_MAX + 1) + (max + 1);
  uint32_t v32 = (UINT8_MAX + 1) + (max + 1);

  // Assignment
  x.c = v8;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint8_t' (aka 'unsigned char') of value 64 (8-bit, unsigned) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to -64 (7-bit bitfield, signed)
  x.c = v16;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint16_t' (aka 'unsigned short') of value 320 (16-bit, unsigned) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to -64 (7-bit bitfield, signed)
  x.c = v32;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 320 (32-bit, unsigned) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to -64 (7-bit bitfield, signed)

  // PrePostIncDec
  x.c = min;
  x.c--;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int' of value -65 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to 63 (7-bit bitfield, signed)
  x.c = min;
  --x.c;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value -65 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to 63 (7-bit bitfield, signed)

  x.c = max;
  x.c++;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int' of value 64 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to -64 (7-bit bitfield, signed)
  x.c = max;
  ++x.c;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value 64 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to -64 (7-bit bitfield, signed)

  x.c = min + 1;
  x.c++;
  x.c = min + 1;
  ++x.c;

  x.c = min + 1;
  x.c--;
  x.c = min + 1;
  --x.c;

  x.c = max - 1;
  x.c++;
  x.c = max - 1;
  ++x.c;

  x.c = max - 1;
  x.c--;
  x.c = max - 1;
  --x.c;

  // Compound assignment
  x.c = 0;
  x.c += max;
  x.c = 0;
  x.c += (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int' of value 64 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to -64 (7-bit bitfield, signed)

  x.c = 0;
  x.c -= (-min);
  x.c = 0;
  x.c -= (-min + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int' of value -65 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to 63 (7-bit bitfield, signed)

  x.c = 1;
  x.c *= max;
  x.c = 1;
  x.c *= (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int' of value 64 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to -64 (7-bit bitfield, signed)
}

void test_d() {
  X x;
  int32_t min = INT16_MIN;
  int32_t max = INT16_MAX;

  uint32_t v32 = max + 1;

  // Assignment
  x.d = v32;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 32768 (32-bit, unsigned) to type 'int32_t' (aka 'int') changed the value to -32768 (16-bit bitfield, signed)

  // PrePostIncDec
  x.d = min;
  x.d--;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -32769 (32-bit, signed) to type 'int32_t' (aka 'int') changed the value to 32767 (16-bit bitfield, signed)
  x.d = min;
  --x.d;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -32769 (32-bit, signed) to type 'int32_t' (aka 'int') changed the value to 32767 (16-bit bitfield, signed)

  x.d = max;
  x.d++;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value 32768 (32-bit, signed) to type 'int32_t' (aka 'int') changed the value to -32768 (16-bit bitfield, signed)
  x.d = max;
  ++x.d;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value 32768 (32-bit, signed) to type 'int32_t' (aka 'int') changed the value to -32768 (16-bit bitfield, signed)

  x.d = min + 1;
  x.d++;
  x.d = min + 1;
  ++x.d;

  x.d = min + 1;
  x.d--;
  x.d = min + 1;
  --x.d;

  x.d = max - 1;
  x.d++;
  x.d = max - 1;
  ++x.d;

  x.d = max - 1;
  x.d--;
  x.d = max - 1;
  --x.d;

  // Compound assignment
  x.d = 0;
  x.d += max;
  x.d = 0;
  x.d += (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int' of value 32768 (32-bit, signed) to type 'int32_t' (aka 'int') changed the value to -32768 (16-bit bitfield, signed)

  x.d = 0;
  x.d -= (-min);
  x.d = 0;
  x.d -= (-min + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int' of value -32769 (32-bit, signed) to type 'int32_t' (aka 'int') changed the value to 32767 (16-bit bitfield, signed)

  x.d = 1;
  x.d *= max;
  x.d = 1;
  x.d *= (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int' of value 32768 (32-bit, signed) to type 'int32_t' (aka 'int') changed the value to -32768 (16-bit bitfield, signed)
}

void test_e() {
  X x;
  uint32_t min = 0;
  uint32_t max = UINT8_MAX;

  uint16_t v16 = max + 1;
  uint32_t v32 = max + 1;

  // Assignment
  x.e = v16;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint16_t' (aka 'unsigned short') of value 256 (16-bit, unsigned) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (8-bit bitfield, unsigned)
  x.e = v32;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 256 (32-bit, unsigned) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (8-bit bitfield, unsigned)

  // PrePostIncDec
  x.e = min;
  x.e--;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int' of value -1 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 255 (8-bit bitfield, unsigned)
  x.e = min;
  --x.e;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value -1 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 255 (8-bit bitfield, unsigned)
  x.e = min + 1;
  x.e--;

  x.e = max;
  x.e++;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int' of value 256 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (8-bit bitfield, unsigned)
  x.e = max;
  ++x.e;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value 256 (32-bit, signed) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (8-bit bitfield, unsigned)
  x.e = max - 1;
  x.e++;

  // Compound assignment
  x.e = 0;
  x.e += max;
  x.e = 0;
  x.e += (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 256 (32-bit, unsigned) to type 'uint8_t' (aka 'unsigned char') changed the value to 0 (8-bit bitfield, unsigned)

  x.e = max;
  x.e -= max;
  x.e = max;
  x.e -= (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type 'uint8_t' (aka 'unsigned char') changed the value to 255 (8-bit bitfield, unsigned)
}

void test_f() {
  X x;
  uint32_t min = 0;
  uint32_t max = UINT16_MAX;

  uint32_t v32 = max + 1;

  // Assignment
  x.f = v32;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 65536 (32-bit, unsigned) to type 'uint16_t' (aka 'unsigned short') changed the value to 0 (16-bit bitfield, unsigned)

  // PrePostIncDec
  x.f = min;
  x.f--;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int' of value -1 (32-bit, signed) to type 'uint16_t' (aka 'unsigned short') changed the value to 65535 (16-bit bitfield, unsigned)
  x.f = min;
  --x.f;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value -1 (32-bit, signed) to type 'uint16_t' (aka 'unsigned short') changed the value to 65535 (16-bit bitfield, unsigned)
  x.f = min + 1;
  x.f--;

  x.f = max;
  x.f++;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int' of value 65536 (32-bit, signed) to type 'uint16_t' (aka 'unsigned short') changed the value to 0 (16-bit bitfield, unsigned)
  x.f = max;
  ++x.f;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value 65536 (32-bit, signed) to type 'uint16_t' (aka 'unsigned short') changed the value to 0 (16-bit bitfield, unsigned)
  x.f = max - 1;
  x.f++;

  // Compound assignment
  x.f = 0;
  x.f += max;
  x.f = 0;
  x.f += (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 65536 (32-bit, unsigned) to type 'uint16_t' (aka 'unsigned short') changed the value to 0 (16-bit bitfield, unsigned)

  x.f = max;
  x.f -= max;
  x.f = max;
  x.f -= (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type 'uint16_t' (aka 'unsigned short') changed the value to 65535 (16-bit bitfield, unsigned)
}

void test_g() {
  X x;
  uint64_t min = 0;
  uint64_t max = UINT32_MAX;

  uint64_t v64 = max + 1;

  // Assignment
  x.g = v64;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint64_t' (aka 'unsigned long{{( long)?}}') of value 4294967296 (64-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 0 (32-bit bitfield, unsigned)

  // PrePostIncDec
  x.g = min;
  x.g--;
  x.g = min;
  --x.g;
  x.g = min + 1;
  x.g--;

  x.g = max;
  x.g++;
  x.g = max;
  ++x.g;
  x.g = max - 1;
  x.g++;

  // Compound assignment
  x.g = 0;
  x.g += max;
  x.g = 0;
  x.g += (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint64_t' (aka 'unsigned long{{( long)?}}') of value 4294967296 (64-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 0 (32-bit bitfield, unsigned)

  x.g = max;
  x.g -= max;
  x.g = max;
  x.g -= (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint64_t' (aka 'unsigned long{{( long)?}}') of value 18446744073709551615 (64-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 4294967295 (32-bit bitfield, unsigned)
}

void test_h() {
  X x;
  int32_t min = INT8_MIN;
  int32_t max = INT8_MAX;

  int16_t v16 = max + 1;
  int32_t v32 = max + 1;

  // Assignment
  x.h = v16;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int16_t' (aka 'short') of value 128 (16-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to -128 (8-bit bitfield, signed)
  x.h = v32;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value 128 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to -128 (8-bit bitfield, signed)

  // PrePostIncDec
  x.h = min;
  x.h--;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int' of value -129 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to 127 (8-bit bitfield, signed)
  x.h = min;
  --x.h;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value -129 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to 127 (8-bit bitfield, signed)
  x.h = min + 1;
  x.h--;

  x.h = max;
  x.h++;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int' of value 128 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to -128 (8-bit bitfield, signed)
  x.h = max;
  ++x.h;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value 128 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to -128 (8-bit bitfield, signed)
  x.h = max - 1;
  x.h++;

  // Compound assignment
  x.h = 0;
  x.h += max;
  x.h = 0;
  x.h += (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int' of value 128 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to -128 (8-bit bitfield, signed)

  x.h = 0;
  x.h -= (-min);
  x.h = 0;
  x.h -= (-min + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int' of value -129 (32-bit, signed) to type 'int8_t' (aka '{{(signed )?}}char') changed the value to 127 (8-bit bitfield, signed)
}

void test_i() {
  X x;
  int32_t min = INT16_MIN;
  int32_t max = INT16_MAX;

  int32_t v32 = max + 1;

  // Assignment
  x.i = v32;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value 32768 (32-bit, signed) to type 'int16_t' (aka 'short') changed the value to -32768 (16-bit bitfield, signed)

  // PrePostIncDec
  x.i = min;
  x.i--;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int' of value -32769 (32-bit, signed) to type 'int16_t' (aka 'short') changed the value to 32767 (16-bit bitfield, signed)
  x.i = min;
  --x.i;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value -32769 (32-bit, signed) to type 'int16_t' (aka 'short') changed the value to 32767 (16-bit bitfield, signed)
  x.i = min + 1;
  x.i--;

  x.i = max;
  x.i++;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int' of value 32768 (32-bit, signed) to type 'int16_t' (aka 'short') changed the value to -32768 (16-bit bitfield, signed)
  x.i = max;
  ++x.i;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:3: runtime error: implicit conversion from type 'int' of value 32768 (32-bit, signed) to type 'int16_t' (aka 'short') changed the value to -32768 (16-bit bitfield, signed)
  x.i = max - 1;
  x.i++;

  // Compound assignment
  x.i = 0;
  x.i += max;
  x.i = 0;
  x.i += (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int' of value 32768 (32-bit, signed) to type 'int16_t' (aka 'short') changed the value to -32768 (16-bit bitfield, signed)

  x.i = 0;
  x.i -= (-min);
  x.i = 0;
  x.i -= (-min + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int' of value -32769 (32-bit, signed) to type 'int16_t' (aka 'short') changed the value to 32767 (16-bit bitfield, signed)
}

void test_j() {
  X x;
  int64_t min = INT32_MIN;
  int64_t max = INT32_MAX;

  int64_t v64 = max + 1;

  // Assignment
  x.j = v64;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int64_t' (aka 'long{{( long)?}}') of value 2147483648 (64-bit, signed) to type 'int32_t' (aka 'int') changed the value to -2147483648 (32-bit bitfield, signed)

  // PrePostIncDec
  x.j = min;
  x.j--;
  x.j = min;
  --x.j;
  x.j = min + 1;
  x.j--;

  x.j = max;
  x.j++;
  x.j = max;
  ++x.j;
  x.j = max - 1;
  x.j++;

  // Compound assignment
  x.j = 0;
  x.j += max;
  x.j = 0;
  x.j += (max + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int64_t' (aka 'long{{( long)?}}') of value 2147483648 (64-bit, signed) to type 'int32_t' (aka 'int') changed the value to -2147483648 (32-bit bitfield, signed)

  x.j = 0;
  x.j -= (-min);
  x.j = 0;
  x.j -= (-min + 1);
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int64_t' (aka 'long{{( long)?}}') of value -2147483649 (64-bit, signed) to type 'int32_t' (aka 'int') changed the value to 2147483647 (32-bit bitfield, signed)
}

void test_k_l() {
  X x;
  int32_t one = 1;
  int32_t neg_one = -1;

  // k
  uint8_t v8 = 2;
  x.k = v8;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint8_t' (aka 'unsigned char') of value 2 (8-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 0 (1-bit bitfield, unsigned)
  x.k = one;
  x.k = neg_one;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -1 (32-bit, signed) to type 'uint32_t' (aka 'unsigned int') changed the value to 1 (1-bit bitfield, unsigned)

  x.k = 0;
  x.k--;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 4294967295 (32-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 1 (1-bit bitfield, unsigned)
  x.k = 1;
  x.k--;

  x.k = 1;
  x.k++;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'uint32_t' (aka 'unsigned int') of value 2 (32-bit, unsigned) to type 'uint32_t' (aka 'unsigned int') changed the value to 0 (1-bit bitfield, unsigned)
  x.k = 0;
  x.k++;

  // l
  x.l = v8;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'uint8_t' (aka 'unsigned char') of value 2 (8-bit, unsigned) to type 'int32_t' (aka 'int') changed the value to 0 (1-bit bitfield, signed)
  x.l = one;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:7: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value 1 (32-bit, signed) to type 'int32_t' (aka 'int') changed the value to -1 (1-bit bitfield, signed)
  x.l = neg_one;

  x.l = 0;
  x.l--;
  x.l = -1;
  x.l--;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value -2 (32-bit, signed) to type 'int32_t' (aka 'int') changed the value to 0 (1-bit bitfield, signed)

  x.l = 0;
  x.l++;
  // CHECK: {{.*}}bitfield-conversion.c:[[@LINE-1]]:6: runtime error: implicit conversion from type 'int32_t' (aka 'int') of value 1 (32-bit, signed) to type 'int32_t' (aka 'int') changed the value to -1 (1-bit bitfield, signed)
  x.l = -1;
  x.l++;
}

void test_m() {
  X x;

  uint8_t v8 = 2;
  x.m = v8;
}

int main() {
  test_a();
  test_b();
  test_c();
  test_d();
  test_e();
  test_f();
  test_g();
  test_h();
  test_i();
  test_j();
  test_k_l();
  test_m();
  return 0;
}
