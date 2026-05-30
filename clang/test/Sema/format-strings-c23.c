// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -ffreestanding -fsyntax-only -verify %s

#include <stdint.h>

int printf(const char *restrict, ...) __attribute__((format(printf, 1, 2)));
int scanf(const char *restrict, ...) __attribute__((format(scanf, 1, 2)));

void t1(int8_t i8, uint8_t u8, int16_t i16, uint16_t u16, int32_t i32,
        uint32_t u32, int64_t i64, uint64_t u64, int_fast8_t if8,
        uint_fast8_t uf8, int_fast16_t if16, uint_fast16_t uf16,
        int_fast32_t if32, uint_fast32_t uf32, int_fast64_t if64,
        uint_fast64_t uf64) {
  printf("%w8d", i8);
  printf("%w8u", u8);
  printf("%w16d", i16);
  printf("%w16u", u16);
  printf("%w32d", i32);
  printf("%w32i", i32);
  printf("%w32u", u32);
  printf("%w32x", u32);
  printf("%w32b", u32);
  printf("%w64d", i64);
  printf("%w64u", u64);
  printf("%wf8d", if8);
  printf("%wf8u", uf8);
  printf("%wf16d", if16);
  printf("%wf16u", uf16);
  printf("%wf32d", if32);
  printf("%wf32u", uf32);
  printf("%wf32B", uf32);
  printf("%wf64d", if64);
  printf("%wf64u", uf64);

  printf("%w32d", 1.0);  // expected-warning{{format specifies type 'int32_t' (aka 'int') but the argument has type 'double'}}
  printf("%w32u", 1.0);  // expected-warning{{format specifies type 'uint32_t' (aka 'unsigned int') but the argument has type 'double'}}
  printf("%wf32d", 1.0); // expected-warning{{format specifies type 'int_fast32_t' (aka 'int') but the argument has type 'double'}}
  printf("%wf32u", 1.0); // expected-warning{{format specifies type 'uint_fast32_t' (aka 'unsigned int') but the argument has type 'double'}}
  printf("%w18446744073709551616d", i32); // expected-warning{{invalid conversion specifier '1'}}
}

void t2(int8_t *i8_ptr, int16_t *i16_ptr, int32_t *i32_ptr,
        int64_t *i64_ptr, int_fast8_t *if8_ptr, int_fast16_t *if16_ptr,
        int_fast32_t *if32_ptr, int_fast64_t *if64_ptr, double *d_ptr) {
  printf("%w8n", i8_ptr);
  printf("%w16n", i16_ptr);
  printf("%w32n", i32_ptr);
  printf("%w64n", i64_ptr);
  printf("%wf8n", if8_ptr);
  printf("%wf16n", if16_ptr);
  printf("%wf32n", if32_ptr);
  printf("%wf64n", if64_ptr);
  printf("%w32n", d_ptr);  // expected-warning{{format specifies type 'int32_t *' (aka 'int *') but the argument has type 'double *'}}
  printf("%wf32n", d_ptr); // expected-warning{{format specifies type 'int_fast32_t *' (aka 'int *') but the argument has type 'double *'}}
}

void t3(int8_t *i8_ptr, uint8_t *u8_ptr, int16_t *i16_ptr,
        uint16_t *u16_ptr, int32_t *i32_ptr, uint32_t *u32_ptr,
        int64_t *i64_ptr, uint64_t *u64_ptr, int_fast8_t *if8_ptr,
        uint_fast8_t *uf8_ptr, int_fast16_t *if16_ptr,
        uint_fast16_t *uf16_ptr, int_fast32_t *if32_ptr,
        uint_fast32_t *uf32_ptr, int_fast64_t *if64_ptr,
        uint_fast64_t *uf64_ptr, double *d_ptr) {
  scanf("%w8d", i8_ptr);
  scanf("%w8u", u8_ptr);
  scanf("%w16d", i16_ptr);
  scanf("%w16u", u16_ptr);
  scanf("%w32d", i32_ptr);
  scanf("%w32i", i32_ptr);
  scanf("%w32u", u32_ptr);
  scanf("%w32x", u32_ptr);
  scanf("%w32b", u32_ptr);
  scanf("%w64d", i64_ptr);
  scanf("%w64u", u64_ptr);
  scanf("%wf8d", if8_ptr);
  scanf("%wf8u", uf8_ptr);
  scanf("%wf16d", if16_ptr);
  scanf("%wf16u", uf16_ptr);
  scanf("%wf32d", if32_ptr);
  scanf("%wf32u", uf32_ptr);
  scanf("%wf64d", if64_ptr);
  scanf("%wf64u", uf64_ptr);

  scanf("%w32d", d_ptr);  // expected-warning{{format specifies type 'int32_t *' (aka 'int *') but the argument has type 'double *'}}
  scanf("%w32u", d_ptr);  // expected-warning{{format specifies type 'uint32_t *' (aka 'unsigned int *') but the argument has type 'double *'}}
  scanf("%wf32d", d_ptr); // expected-warning{{format specifies type 'int_fast32_t *' (aka 'int *') but the argument has type 'double *'}}
  scanf("%wf32u", d_ptr); // expected-warning{{format specifies type 'uint_fast32_t *' (aka 'unsigned int *') but the argument has type 'double *'}}
}

void t4(const char *fmt) __attribute__((format_matches(printf, 1, "%w32d"))); // expected-note{{comparing with this specifier}}
void t5(void) {
  t4("%w32d");
  t4("%w64d"); // expected-warning{{format specifier 'w64d' is incompatible with 'w32d'}}
}
