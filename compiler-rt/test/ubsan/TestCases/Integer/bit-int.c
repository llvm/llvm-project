// REQUIRES: x86_64-target-arch
// REQUIRES: !windows
// RUN: %clang -Wno-constant-conversion -Wno-array-bounds -Wno-division-by-zero -Wno-shift-negative-value -Wno-shift-count-negative -Wno-int-to-pointer-cast -O0 -fsanitize=array-bounds,float-cast-overflow,implicit-integer-sign-change,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation,integer-divide-by-zero,pointer-overflow,shift-base,shift-exponent,signed-integer-overflow,unsigned-integer-overflow,unsigned-shift-base,vla-bound %s -o %t1 && %run %t1 2>&1 | FileCheck %s

#include <stdint.h>
#include <stdio.h>

uint32_t float_divide_by_zero() {
  float f = 1.0f / 0.0f;
  _BitInt(37) r = (_BitInt(37))f;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:19: runtime error: inf is outside the range of representable values of type
  return r;
}

uint32_t integer_divide_by_zero() __attribute__((no_sanitize("memory"))) {
  _BitInt(37) x = 1 / 0;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:21: runtime error: division by zero
  return x;
}

uint32_t implicit_unsigned_integer_truncation() {
  unsigned _BitInt(37) x = 2U;
  x += float_divide_by_zero();
  x += integer_divide_by_zero();
  x = x + 0xFFFFFFFFFFFFFFFFULL;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:9: runtime error: unsigned integer overflow:
  uint32_t r = x & 0xFFFFFFFF;
  return r;
}

uint32_t pointer_overflow() __attribute__((no_sanitize("address"))) {
  _BitInt(37) *x = (_BitInt(37) *)1;
  _BitInt(37) *y = x - 1;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:22: runtime error: pointer index expression with base
  uint32_t r = *(_BitInt(37) *)&y;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:16: runtime error: implicit conversion from type
  return r;
}

uint32_t vla_bound(_BitInt(37) x) {
  _BitInt(37) a[x - 1];
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:17: runtime error: variable length array bound evaluates to non-positive value
  return 0;
}

uint32_t unsigned_shift_base() {
  unsigned _BitInt(37) x = ~0U << 1;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:32: runtime error: left shift of 4294967295 by 1 places cannot be represented in type
  return x;
}

uint32_t array_bounds() {
  _BitInt(37) x[4];
  _BitInt(37) y = x[10];
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:19: runtime error: index 10 out of bounds for type
  return (uint32_t)y;
}

uint32_t float_cast_overflow() {
  float a = 100000000.0f;
  _BitInt(7) b = (_BitInt(7))a;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:18: runtime error: 1e+08 is outside the range of representable values of type
  return b;
}

uint32_t implicit_integer_sign_change(unsigned _BitInt(37) x) {
  _BitInt(37) r = x;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:19: runtime error: implicit conversion from type '{{[^']+}}' of value
  return r & 0xFFFFFFFF;
}

_BitInt(13) implicit_signed_integer_truncation() {
#ifdef __SIZEOF_INT128__
  _BitInt(73) x = (_BitInt(73)) ~((~0UL) >> 1);
#else
  uint32_t x = 0x7FFFFFFFUL;
#endif
  return x;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:10: runtime error: implicit conversion from type
}

_BitInt(37) shift_exponent() __attribute__((no_sanitize("memory"))) {
  _BitInt(37) x = 1 << (-1);
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:21: runtime error: shift exponent -1 is negative
  return x;
}

_BitInt(37) shift_base() __attribute__((no_sanitize("memory"))) {
  _BitInt(37) x = (-1) << 1;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:24: runtime error: left shift of negative value -1
  return x;
}

uint32_t negative_shift1(unsigned _BitInt(37) x)
    __attribute__((no_sanitize("memory"))) {
  _BitInt(9) c = -2;
  return x >> c;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:12: runtime error: shift exponent -2 is negative
}

uint32_t negative_shift2(unsigned _BitInt(37) x)
    __attribute__((no_sanitize("memory"))) {
  _BitInt(17) c = -2;
  return x >> c;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:12: runtime error: shift exponent -2 is negative
}

uint32_t negative_shift3(unsigned _BitInt(37) x)
    __attribute__((no_sanitize("memory"))) {
  _BitInt(34) c = -2;
  return x >> c;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:12: runtime error: shift exponent -2 is negative
}

uint32_t negative_shift4(unsigned _BitInt(37) x)
    __attribute__((no_sanitize("memory"))) {
  int64_t c = -2;
  return x >> c;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:12: runtime error: shift exponent -2 is negative
}

uint32_t negative_shift5(unsigned _BitInt(37) x)
    __attribute__((no_sanitize("memory"))) {
#ifdef __SIZEOF_INT128__
  _BitInt(68) c = -2;
#else
  // We cannot check BitInt values > 64 without int128_t support
  _BitInt(48) c = -2;
#endif
  return x >> c;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:12: runtime error: shift exponent -2 is negative
}

uint32_t unsigned_integer_overflow() __attribute__((no_sanitize("memory"))) {
  unsigned _BitInt(37) x = ~0U;
  ++x;
  return x;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:10: runtime error: implicit conversion from type
}

// In this test no run-time overflow expected, so no diagnostics here, but should be a conversion error from the negative number on return.
uint32_t signed_integer_overflow() __attribute__((no_sanitize("memory"))) {
  _BitInt(37) x = (_BitInt(37)) ~((0x8FFFFFFFFFFFFFFFULL) >> 1);
  --x;
  return x;
  // CHECK: {{.*}}bit-int.c:[[#@LINE-1]]:10: runtime error: implicit conversion from type
}

int main(int argc, char **argv) {
  // clang-format off
  uint64_t result =
      1ULL +
      implicit_unsigned_integer_truncation() +
      pointer_overflow() +
      vla_bound(argc) +
      unsigned_shift_base() +
      (uint32_t)array_bounds() +
      float_cast_overflow() +
      implicit_integer_sign_change((unsigned _BitInt(37))(argc - 2)) +
      (uint64_t)implicit_signed_integer_truncation() +
      shift_exponent() +
      (uint32_t)shift_base() +
      negative_shift1(5) +
      negative_shift2(5) +
      negative_shift3(5) +
      negative_shift4(5) +
      negative_shift5(5) +
      unsigned_integer_overflow() +
      signed_integer_overflow();
  // clang-format on
  printf("%u\n", (uint32_t)(result & 0xFFFFFFFF));
}
