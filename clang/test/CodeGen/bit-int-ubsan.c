// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -Wno-constant-conversion -Wno-array-bounds -Wno-division-by-zero -Wno-shift-negative-value -Wno-shift-count-negative -Wno-int-to-pointer-cast -fsanitize=array-bounds,enum,float-cast-overflow,integer-divide-by-zero,implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change,unsigned-integer-overflow,signed-integer-overflow,shift-base,shift-exponent -O0 -emit-llvm -o - %s | FileCheck %s

// The runtime test checking the _BitInt ubsan feature is located in compiler-rt/test/ubsan/TestCases/Integer/bit-int.c

typedef unsigned int uint32_t;
uint32_t float_divide_by_zero() {
  float f = 1.0f / 0.0f;
  // CHECK: constant { i16, i16, [8 x i8] } { i16 1, i16 32, [8 x i8] c"'float'\00" }
  _BitInt(37) r = (_BitInt(37))f;
  // CHECK: constant { i16, i16, [20 x i8] } { i16 2, i16 13, [20 x i8] c"'_BitInt(37)'\00%\00\00\00\00\00" }
  return r;
}

uint32_t integer_divide_by_zero() __attribute__((no_sanitize("memory"))) {
  _BitInt(37) x = 1 / 0;
  // CHECK: constant { i16, i16, [32 x i8] } { i16 0, i16 10, [32 x i8] c"'uint32_t' (aka 'unsigned int')\00" }
  return x;
}

uint32_t implicit_unsigned_integer_truncation() {
  unsigned _BitInt(37) x = 2U;
  x += float_divide_by_zero();
  x += integer_divide_by_zero();
  x = x + 0xFFFFFFFFFFFFFFFFULL;
  // CHECK: constant { i16, i16, [23 x i8] } { i16 0, i16 12, [23 x i8] c"'unsigned _BitInt(37)'\00" }
  uint32_t r = x & 0xFFFFFFFF;
  return r;
}

uint32_t array_bounds() {
  _BitInt(37) x[4];
  _BitInt(37) y = x[10];
  // CHECK: constant { i16, i16, [17 x i8] } { i16 -1, i16 0, [17 x i8] c"'_BitInt(37)[4]'\00" }
  return (uint32_t)y;
}

uint32_t float_cast_overflow() {
  float a = 100000000.0f;
  _BitInt(7) b = (_BitInt(7))a;
  // CHECK: constant { i16, i16, [19 x i8] } { i16 2, i16 7, [19 x i8] c"'_BitInt(7)'\00\07\00\00\00\00\00" }
  return b;
}

_BitInt(13) implicit_signed_integer_truncation() {
  _BitInt(73) x = (_BitInt(73)) ~((~0UL) >> 1);
  return x;
  // CHECK: constant { i16, i16, [20 x i8] } { i16 2, i16 {{([[:xdigit:]]{2})}}, [20 x i8] c"'_BitInt(73)'\00I\00\00\00\00\00" }
  // CHECK: constant { i16, i16, [20 x i8] } { i16 2, i16 9, [20 x i8] c"'_BitInt(13)'\00\0D\00\00\00\00\00" }
}

uint32_t negative_shift1(unsigned _BitInt(37) x)
    __attribute__((no_sanitize("memory"))) {
  _BitInt(9) c = -2;
  return x >> c;
  // CHECK: constant { i16, i16, [19 x i8] } { i16 2, i16 9, [19 x i8] c"'_BitInt(9)'\00\09\00\00\00\00\00" }
}

uint32_t negative_shift2(unsigned _BitInt(37) x)
    __attribute__((no_sanitize("memory"))) {
  _BitInt(17) c = -2;
  return x >> c;
  // CHECK: constant { i16, i16, [20 x i8] } { i16 2, i16 11, [20 x i8] c"'_BitInt(17)'\00\11\00\00\00\00\00" }
}

uint32_t negative_shift3(unsigned _BitInt(37) x)
    __attribute__((no_sanitize("memory"))) {
  _BitInt(34) c = -2;
  return x >> c;
  // CHECK: constant { i16, i16, [20 x i8] } { i16 2, i16 13, [20 x i8] c"'_BitInt(34)'\00\22\00\00\00\00\00" }
}

uint32_t negative_shift5(unsigned _BitInt(37) x)
    __attribute__((no_sanitize("memory"))) {
  _BitInt(68) c = -2;
  return x >> c;
  // CHECK: constant { i16, i16, [20 x i8] } { i16 2, i16 {{([[:xdigit:]]{2})}}, [20 x i8] c"'_BitInt(68)'\00D\00\00\00\00\00" }
}
