// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -isystem %S/Inputs -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -isystem %S/Inputs -fsyntax-only -verify %s -fexperimental-new-constant-interpreter
#include <stdbit.h>

// LE unsigned: bytes ordered LSB-first (index 0 = byte 0 of the value)
// alignas(8) so these buffers also satisfy the stdc_load8_aligned_* variants.
alignas(8) static const unsigned char le8_u[]  = {0xAB};
alignas(8) static const unsigned char le16_u[] = {0x34, 0x12};
alignas(8) static const unsigned char le32_u[] = {0x78, 0x56, 0x34, 0x12};
alignas(8) static const unsigned char le64_u[] = {0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12};

_Static_assert(stdc_load8_leu8(le8_u)   == (__UINT_LEAST8_TYPE__)0xAB,               "");
_Static_assert(stdc_load8_leu16(le16_u) == (__UINT_LEAST16_TYPE__)0x1234,            "");
_Static_assert(stdc_load8_leu32(le32_u) == (__UINT_LEAST32_TYPE__)0x12345678U,       "");
_Static_assert(stdc_load8_leu64(le64_u) == (__UINT_LEAST64_TYPE__)0x123456789ABCDEF0ULL, "");

// BE unsigned: bytes ordered MSB-first (index 0 = highest byte)
alignas(8) static const unsigned char be8_u[]  = {0xAB};
alignas(8) static const unsigned char be16_u[] = {0x12, 0x34};
alignas(8) static const unsigned char be32_u[] = {0x12, 0x34, 0x56, 0x78};
alignas(8) static const unsigned char be64_u[] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

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
alignas(8) static const unsigned char le8_s[]  = {0x80};
alignas(8) static const unsigned char le16_s[] = {0x80, 0xFF};
alignas(8) static const unsigned char le32_s[] = {0x80, 0xFF, 0xFF, 0xFF};
alignas(8) static const unsigned char le64_s[] = {0x80, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

_Static_assert(stdc_load8_les8(le8_s)   == (__INT_LEAST8_TYPE__)-128,  "");
_Static_assert(stdc_load8_les16(le16_s) == (__INT_LEAST16_TYPE__)-128, "");
_Static_assert(stdc_load8_les32(le32_s) == (__INT_LEAST32_TYPE__)-128, "");
_Static_assert(stdc_load8_les64(le64_s) == (__INT_LEAST64_TYPE__)-128, "");

// BE signed
alignas(8) static const unsigned char be8_s[]  = {0x80};
alignas(8) static const unsigned char be16_s[] = {0xFF, 0x80};
alignas(8) static const unsigned char be32_s[] = {0xFF, 0xFF, 0xFF, 0x80};
alignas(8) static const unsigned char be64_s[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x80};

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

// N=64 signed boundary: result == 2^63 is not < 2^63, so it wraps to
// result - 2^64 == INT64_MIN, not a positive value.
static const unsigned char le64_min[] = {0, 0, 0, 0, 0, 0, 0, 0x80};
static const unsigned char be64_min[] = {0x80, 0, 0, 0, 0, 0, 0, 0};
_Static_assert(stdc_load8_les64(le64_min) == (__INT_LEAST64_TYPE__)(-9223372036854775807LL - 1), "");
_Static_assert(stdc_load8_bes64(be64_min) == (__INT_LEAST64_TYPE__)(-9223372036854775807LL - 1), "");

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
  __UINT_LEAST8_TYPE__ x = stdc_load8_leu8(0);       // expected-warning{{null passed to a callee that requires a non-null argument}}
  __UINT_LEAST8_TYPE__ y = stdc_load8_leu8(nullptr);  // expected-warning{{null passed to a callee that requires a non-null argument}}
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
constexpr __UINT_LEAST32_TYPE__ null_ce_nullptr = stdc_load8_leu32(nullptr); // expected-error{{must be initialized by a constant expression}} expected-note{{read of dereferenced null pointer is not allowed in a constant expression}}

constexpr unsigned char one[] = {0x42};
constexpr __UINT_LEAST8_TYPE__ oob_past_end = stdc_load8_leu8(one + 1); // expected-error{{must be initialized by a constant expression}} expected-note{{cannot refer to element 1 of array of 1 element in a constant expression}}

// aligned_* variants require the pointer to be aligned to the result type.
alignas(8) constexpr unsigned char align_buf[9] = {0, 0x78, 0x56, 0x34, 0x12, 0, 0, 0, 0};

constexpr __UINT_LEAST16_TYPE__ misaligned16 = stdc_load8_aligned_leu16(align_buf + 1); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_leu16' requires a pointer aligned to 2 bytes, but the given pointer is only aligned to 1 byte}}
constexpr __UINT_LEAST32_TYPE__ misaligned32 = stdc_load8_aligned_leu32(align_buf + 1); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_leu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 1 byte}}
constexpr __UINT_LEAST64_TYPE__ misaligned64 = stdc_load8_aligned_leu64(align_buf + 1); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_leu64' requires a pointer aligned to 8 bytes, but the given pointer is only aligned to 1 byte}}

// Offset 2 from an 8-byte aligned base is only 2-byte aligned, enough for a
// 16-bit load but not a 32-bit one.
constexpr __UINT_LEAST32_TYPE__ misaligned32_half = stdc_load8_aligned_leu32(align_buf + 2); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_leu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 2 bytes}}

// Offset 4 is still 4-byte aligned, enough for a 16-bit (2-byte) load.
// align_buf[4..5] == {0x12, 0x00}, so LE u16 == 0x0012.
constexpr __UINT_LEAST16_TYPE__ partially_aligned16 = stdc_load8_aligned_leu16(align_buf + 4);
static_assert(partially_aligned16 == 0x0012, "");

constexpr unsigned char noalign_buf[8] = {0x78, 0x56, 0x34, 0x12, 0, 0, 0, 0};
constexpr __UINT_LEAST8_TYPE__  noalign_ok   = stdc_load8_aligned_leu8(noalign_buf);
static_assert(noalign_ok == 0x78, "");
constexpr __UINT_LEAST16_TYPE__ noalign_fail16 = stdc_load8_aligned_leu16(noalign_buf); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_leu16' requires a pointer aligned to 2 bytes, but the given pointer is only aligned to 1 byte}}
constexpr __UINT_LEAST32_TYPE__ noalign_fail32 = stdc_load8_aligned_leu32(noalign_buf); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_leu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 1 byte}}

void test_block_scope_vardecl(void) {
  alignas(8) constexpr unsigned char blk_buf[4] = {0x78, 0x56, 0x34, 0x12};
  static_assert(stdc_load8_aligned_leu32(blk_buf) == 0x12345678U, "");

  constexpr unsigned char blk_noalign[4] = {0x78, 0x56, 0x34, 0x12};
  constexpr __UINT_LEAST32_TYPE__ blk_fail = stdc_load8_aligned_leu32(blk_noalign); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_leu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 1 byte}}
}

constexpr __UINT_LEAST8_TYPE__ strlit_ok = stdc_load8_aligned_leu8((const unsigned char *)u8"\xAB");
static_assert(strlit_ok == (__UINT_LEAST8_TYPE__)0xAB, "");

// Multi-byte loads from a string literal: string-literal pointers have no
// Descriptor, so this exercises the byte-loop/indexing path for that
// pointer kind, LE and BE.
constexpr __UINT_LEAST32_TYPE__ strlit_plain_le = stdc_load8_leu32((const unsigned char *)u8"\x78\x56\x34\x12");
static_assert(strlit_plain_le == 0x12345678U, "");

constexpr __INT_LEAST16_TYPE__ strlit_plain_be = stdc_load8_bes16((const unsigned char *)u8"\xFF\x80");
static_assert(strlit_plain_be == -128, "");

constexpr __UINT_LEAST16_TYPE__ strlit_fail = stdc_load8_aligned_leu16((const unsigned char *)u8"\x34\x12"); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_leu16' requires a pointer aligned to 2 bytes, but the given pointer is only aligned to 1 byte}}

constexpr __UINT_LEAST32_TYPE__ complit_fail = stdc_load8_aligned_leu32((unsigned char[4]){0x78, 0x56, 0x34, 0x12}); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_leu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 1 byte}}

typedef unsigned char AlignedBuf8[8] __attribute__((aligned(8)));

constexpr __UINT_LEAST32_TYPE__ complit_offset_fail = stdc_load8_aligned_leu32((AlignedBuf8){0, 0, 0x78, 0x56, 0x34, 0x12} + 2); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_leu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 2 bytes}}

alignas(8) constexpr unsigned char align_buf_be[9] = {0, 0x12, 0x34, 0x56, 0x78, 0, 0, 0, 0};

constexpr __UINT_LEAST16_TYPE__ misaligned16_be = stdc_load8_aligned_beu16(align_buf_be + 1); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_beu16' requires a pointer aligned to 2 bytes, but the given pointer is only aligned to 1 byte}}
constexpr __UINT_LEAST32_TYPE__ misaligned32_be = stdc_load8_aligned_beu32(align_buf_be + 1); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_beu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 1 byte}}
constexpr __UINT_LEAST64_TYPE__ misaligned64_be = stdc_load8_aligned_beu64(align_buf_be + 1); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_beu64' requires a pointer aligned to 8 bytes, but the given pointer is only aligned to 1 byte}}

constexpr __UINT_LEAST16_TYPE__ partially_aligned16_be = stdc_load8_aligned_beu16(align_buf_be + 4);
static_assert(partially_aligned16_be == 0x7800, "");

alignas(8) constexpr unsigned char align_buf_les[9] = {0, 0x80, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0};
constexpr __INT_LEAST32_TYPE__ misaligned32_les = stdc_load8_aligned_les32(align_buf_les + 1); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_les32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 1 byte}}

alignas(8) constexpr unsigned char align_buf_bes[9] = {0, 0xFF, 0xFF, 0xFF, 0x80, 0, 0, 0, 0};
constexpr __INT_LEAST32_TYPE__ misaligned32_bes = stdc_load8_aligned_bes32(align_buf_bes + 1); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_bes32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 1 byte}}

alignas(8) constexpr unsigned char onepastend_buf[4] = {0x78, 0x56, 0x34, 0x12};
constexpr __UINT_LEAST32_TYPE__ onepastend_aligned = stdc_load8_aligned_leu32(onepastend_buf + 4); // expected-error{{must be initialized by a constant expression}} expected-note{{cannot refer to element 7 of array of 4 elements in a constant expression}}

alignas(8) constexpr unsigned char onepastend_misaligned_buf[5] = {0, 0x78, 0x56, 0x34, 0x12};
constexpr __UINT_LEAST32_TYPE__ onepastend_misaligned = stdc_load8_aligned_leu32(onepastend_misaligned_buf + 5); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_leu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 1 byte}}

struct FieldOffset { unsigned char pad; unsigned char data[8]; };
alignas(8) constexpr struct FieldOffset field_s = {0, {0x78, 0x56, 0x34, 0x12}};
constexpr __UINT_LEAST32_TYPE__ field_fail = stdc_load8_aligned_leu32(field_s.data); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_leu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 1 byte}}

alignas(8) constexpr struct FieldOffset field_arr[2] = {{0}, {0, {0x78, 0x56, 0x34, 0x12}}};
constexpr __UINT_LEAST32_TYPE__ field_arr_fail = stdc_load8_aligned_leu32(field_arr[1].data); // expected-error{{must be initialized by a constant expression}} expected-note{{'stdc_load8_aligned_leu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 2 bytes}}

union FieldUnion { unsigned char c; unsigned char data[8]; };
alignas(8) constexpr union FieldUnion field_u = {.data = {0x78, 0x56, 0x34, 0x12}};
constexpr __UINT_LEAST32_TYPE__ union_ok = stdc_load8_aligned_leu32(field_u.data);
static_assert(union_ok == 0x12345678U, "");

constexpr unsigned char scalar_ok = 0x42;
constexpr __UINT_LEAST8_TYPE__ scalar_aligned_ok = stdc_load8_aligned_leu8(&scalar_ok);
static_assert(scalar_aligned_ok == 0x42, "");

alignas(4) constexpr unsigned char scalar_aligned_oob = 0x42;
constexpr __UINT_LEAST32_TYPE__ oob_scalar_aligned = stdc_load8_aligned_leu32(&scalar_aligned_oob); // expected-error{{must be initialized by a constant expression}} expected-note{{cannot refer to element 3 of non-array object in a constant expression}}

void test_block_scope_scalar(void) {
  constexpr unsigned char local_scalar = 0x42;
  static_assert(stdc_load8_aligned_leu8(&local_scalar) == 0x42, "");
}
