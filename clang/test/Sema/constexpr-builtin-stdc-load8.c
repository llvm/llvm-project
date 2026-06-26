// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -isystem %S/Inputs -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -isystem %S/Inputs -fsyntax-only -verify %s -fexperimental-new-constant-interpreter
#include <stdbit.h>

// LE unsigned: bytes ordered LSB-first (index 0 = byte 0 of the value)
static const unsigned char le8_u[]  = {0xAB};
static const unsigned char le16_u[] = {0x34, 0x12};
static const unsigned char le32_u[] = {0x78, 0x56, 0x34, 0x12};
static const unsigned char le64_u[] = {0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12};

_Static_assert(stdc_load8_leu8(le8_u)   == (__UINT_LEAST8_TYPE__)0xAB,               "");
_Static_assert(stdc_load8_leu16(le16_u) == (__UINT_LEAST16_TYPE__)0x1234,            "");
_Static_assert(stdc_load8_leu32(le32_u) == (__UINT_LEAST32_TYPE__)0x12345678U,       "");
_Static_assert(stdc_load8_leu64(le64_u) == (__UINT_LEAST64_TYPE__)0x123456789ABCDEF0ULL, "");

// BE unsigned: bytes ordered MSB-first (index 0 = highest byte)
static const unsigned char be8_u[]  = {0xAB};
static const unsigned char be16_u[] = {0x12, 0x34};
static const unsigned char be32_u[] = {0x12, 0x34, 0x56, 0x78};
static const unsigned char be64_u[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

_Static_assert(stdc_load8_beu8(be8_u)   == (__UINT_LEAST8_TYPE__)0xAB,               "");
_Static_assert(stdc_load8_beu16(be16_u) == (__UINT_LEAST16_TYPE__)0x1234,            "");
_Static_assert(stdc_load8_beu32(be32_u) == (__UINT_LEAST32_TYPE__)0x12345678U,       "");
_Static_assert(stdc_load8_beu64(be64_u) == (__UINT_LEAST64_TYPE__)0x123456789ABCDEF0ULL, "");

// Aligned LE unsigned
_Static_assert(stdc_load8_aligned_leu8(le8_u)   == (__UINT_LEAST8_TYPE__)0xAB,               "");
_Static_assert(stdc_load8_aligned_leu16(le16_u) == (__UINT_LEAST16_TYPE__)0x1234,            "");
_Static_assert(stdc_load8_aligned_leu32(le32_u) == (__UINT_LEAST32_TYPE__)0x12345678U,       "");
_Static_assert(stdc_load8_aligned_leu64(le64_u) == (__UINT_LEAST64_TYPE__)0x123456789ABCDEF0ULL, "");

// Aligned BE unsigned
_Static_assert(stdc_load8_aligned_beu8(be8_u)   == (__UINT_LEAST8_TYPE__)0xAB,               "");
_Static_assert(stdc_load8_aligned_beu16(be16_u) == (__UINT_LEAST16_TYPE__)0x1234,            "");
_Static_assert(stdc_load8_aligned_beu32(be32_u) == (__UINT_LEAST32_TYPE__)0x12345678U,       "");
_Static_assert(stdc_load8_aligned_beu64(be64_u) == (__UINT_LEAST64_TYPE__)0x123456789ABCDEF0ULL, "");

// LE signed: 0x80 in u8 = -128 as s8; {0x80, 0xFF} as u16 = -128 as s16
static const unsigned char le8_s[]  = {0x80};
static const unsigned char le16_s[] = {0x80, 0xFF};
static const unsigned char le32_s[] = {0x80, 0xFF, 0xFF, 0xFF};
static const unsigned char le64_s[] = {0x80, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

_Static_assert(stdc_load8_les8(le8_s)   == (__INT_LEAST8_TYPE__)-128,  "");
_Static_assert(stdc_load8_les16(le16_s) == (__INT_LEAST16_TYPE__)-128, "");
_Static_assert(stdc_load8_les32(le32_s) == (__INT_LEAST32_TYPE__)-128, "");
_Static_assert(stdc_load8_les64(le64_s) == (__INT_LEAST64_TYPE__)-128, "");

// BE signed
static const unsigned char be8_s[]  = {0x80};
static const unsigned char be16_s[] = {0xFF, 0x80};
static const unsigned char be32_s[] = {0xFF, 0xFF, 0xFF, 0x80};
static const unsigned char be64_s[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x80};

_Static_assert(stdc_load8_bes8(be8_s)   == (__INT_LEAST8_TYPE__)-128,  "");
_Static_assert(stdc_load8_bes16(be16_s) == (__INT_LEAST16_TYPE__)-128, "");
_Static_assert(stdc_load8_bes32(be32_s) == (__INT_LEAST32_TYPE__)-128, "");
_Static_assert(stdc_load8_bes64(be64_s) == (__INT_LEAST64_TYPE__)-128, "");

// Aligned LE signed
_Static_assert(stdc_load8_aligned_les8(le8_s)   == (__INT_LEAST8_TYPE__)-128,  "");
_Static_assert(stdc_load8_aligned_les16(le16_s) == (__INT_LEAST16_TYPE__)-128, "");
_Static_assert(stdc_load8_aligned_les32(le32_s) == (__INT_LEAST32_TYPE__)-128, "");
_Static_assert(stdc_load8_aligned_les64(le64_s) == (__INT_LEAST64_TYPE__)-128, "");

// Aligned BE signed
_Static_assert(stdc_load8_aligned_bes8(be8_s)   == (__INT_LEAST8_TYPE__)-128,  "");
_Static_assert(stdc_load8_aligned_bes16(be16_s) == (__INT_LEAST16_TYPE__)-128, "");
_Static_assert(stdc_load8_aligned_bes32(be32_s) == (__INT_LEAST32_TYPE__)-128, "");
_Static_assert(stdc_load8_aligned_bes64(be64_s) == (__INT_LEAST64_TYPE__)-128, "");

// Positive signed round-trip
static const unsigned char le16_pos[] = {0x01, 0x00};
static const unsigned char be16_pos[] = {0x00, 0x01};
_Static_assert(stdc_load8_les16(le16_pos) == 1, "");
_Static_assert(stdc_load8_bes16(be16_pos) == 1, "");

// constexpr variable declarations require constexpr arrays as the source
constexpr unsigned char cx_le32[] = {0x78, 0x56, 0x34, 0x12};
constexpr unsigned char cx_be32[] = {0x12, 0x34, 0x56, 0x78};
constexpr unsigned char cx_le16_s[] = {0x80, 0xFF};
constexpr unsigned char cx_be16_s[] = {0xFF, 0x80};

constexpr __UINT_LEAST32_TYPE__ u32_le = stdc_load8_leu32(cx_le32);
_Static_assert(u32_le == 0x12345678U, "");

constexpr __UINT_LEAST32_TYPE__ u32_be = stdc_load8_beu32(cx_be32);
_Static_assert(u32_be == 0x12345678U, "");

constexpr __INT_LEAST16_TYPE__ s16_le = stdc_load8_les16(cx_le16_s);
_Static_assert(s16_le == -128, "");

constexpr __INT_LEAST16_TYPE__ s16_be = stdc_load8_bes16(cx_be16_s);
_Static_assert(s16_be == -128, "");

// Null pointer is rejected (NonNull attribute)
void test_null(void) {
  __UINT_LEAST8_TYPE__ x = stdc_load8_leu8(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}

// Wrong pointer types are rejected by the type system at the call site.
void test_wrong_types(void) {
  const int int_arr[] = {0};
  const unsigned int uint_arr[] = {0};
  const char char_arr[] = "A";

  (void)stdc_load8_leu32(int_arr);  // expected-error{{incompatible pointer types}} expected-note@Inputs/stdbit.h:*{{passing argument to parameter here}}
  (void)stdc_load8_leu32(uint_arr); // expected-error{{incompatible pointer types}} expected-note@Inputs/stdbit.h:*{{passing argument to parameter here}}
  (void)stdc_load8_leu16(char_arr); // expected-warning{{converts between pointers to integer types}} expected-note@Inputs/stdbit.h:*{{passing argument to parameter here}}
}

// Negative: out-of-bounds, scalar, and null.
constexpr unsigned char small[] = {0x01, 0x02};
constexpr __UINT_LEAST32_TYPE__ oob_load = stdc_load8_leu32(small); // expected-error{{must be initialized by a constant expression}} expected-note{{cannot refer to element 3 of array of 2 elements in a constant expression}}
constexpr __UINT_LEAST32_TYPE__ oob_mid  = stdc_load8_leu32(small + 1); // expected-error{{must be initialized by a constant expression}} expected-note{{cannot refer to element 4 of array of 2 elements in a constant expression}}

constexpr unsigned char scalar_byte = 0x42;
constexpr __UINT_LEAST32_TYPE__ oob_scalar = stdc_load8_leu32(&scalar_byte); // expected-error{{must be initialized by a constant expression}} expected-note{{cannot refer to element 3 of non-array object in a constant expression}}

constexpr __UINT_LEAST32_TYPE__ null_ce = stdc_load8_leu32((const unsigned char *)0); // expected-error{{must be initialized by a constant expression}} expected-note{{read of dereferenced null pointer is not allowed in a constant expression}}

constexpr unsigned char one[] = {0x42};
constexpr __UINT_LEAST8_TYPE__ oob_past_end = stdc_load8_leu8(one + 1); // expected-error{{must be initialized by a constant expression}} expected-note{{cannot refer to element 1 of array of 1 element in a constant expression}}
