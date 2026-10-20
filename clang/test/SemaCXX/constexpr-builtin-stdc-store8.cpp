// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=ref,both -std=c++20 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=expected,both -std=c++20 %s -fexperimental-new-constant-interpreter

// Constant evaluation of the C2y stdc_store8_* functions through their
// __builtin_ spellings. Mutation in a constant expression needs a constexpr
// function, so this is exercised as C++.

// Store VAL with FN into a zeroed, 8-byte aligned buffer at offset 0 and check
// that exactly the listed bytes were written (in order) and nothing else.
#define STORE_TEST(NAME, FN, VAL, ...)                                         \
  constexpr bool NAME() {                                                      \
    alignas(8) unsigned char buf[8] = {};                                      \
    FN(VAL, buf);                                                              \
    const unsigned char expected[] = {__VA_ARGS__};                            \
    unsigned i = 0;                                                            \
    for (; i < sizeof(expected); ++i)                                          \
      if (buf[i] != expected[i])                                               \
        return false;                                                          \
    for (; i < sizeof(buf); ++i)                                               \
      if (buf[i] != 0)                                                         \
        return false;                                                          \
    return true;                                                               \
  }                                                                            \
  static_assert(NAME(), #NAME);


// Unaligned variants, all widths, both byte orders, both signednesses.
STORE_TEST(t_leu8, __builtin_stdc_store8_leu8, 0xA5, 0xa5)
STORE_TEST(t_beu8, __builtin_stdc_store8_beu8, 0xA5, 0xa5)
STORE_TEST(t_leu16, __builtin_stdc_store8_leu16, 0xABCD, 0xcd, 0xab)
STORE_TEST(t_beu16, __builtin_stdc_store8_beu16, 0xABCD, 0xab, 0xcd)
STORE_TEST(t_leu32, __builtin_stdc_store8_leu32, 0x12345678u, 0x78, 0x56, 0x34, 0x12)
STORE_TEST(t_beu32, __builtin_stdc_store8_beu32, 0x12345678u, 0x12, 0x34, 0x56, 0x78)
STORE_TEST(t_leu64, __builtin_stdc_store8_leu64, 0x1122334455667788ull, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11)
STORE_TEST(t_beu64, __builtin_stdc_store8_beu64, 0x1122334455667788ull, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88)
STORE_TEST(t_les8, __builtin_stdc_store8_les8, -128, 0x80)
STORE_TEST(t_bes8, __builtin_stdc_store8_bes8, -128, 0x80)
STORE_TEST(t_les16, __builtin_stdc_store8_les16, -2, 0xfe, 0xff)
STORE_TEST(t_bes16, __builtin_stdc_store8_bes16, -2, 0xff, 0xfe)
STORE_TEST(t_les32, __builtin_stdc_store8_les32, -2, 0xfe, 0xff, 0xff, 0xff)
STORE_TEST(t_bes32, __builtin_stdc_store8_bes32, -2, 0xff, 0xff, 0xff, 0xfe)
STORE_TEST(t_les64, __builtin_stdc_store8_les64, (-9223372036854775807ll - 1), 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x80)
STORE_TEST(t_bes64, __builtin_stdc_store8_bes64, (-9223372036854775807ll - 1), 0x80, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0)

// Aligned variants, all widths, both byte orders, both signednesses.
STORE_TEST(t_aligned_leu8, __builtin_stdc_store8_aligned_leu8, 0xA5, 0xa5)
STORE_TEST(t_aligned_beu8, __builtin_stdc_store8_aligned_beu8, 0xA5, 0xa5)
STORE_TEST(t_aligned_leu16, __builtin_stdc_store8_aligned_leu16, 0xABCD, 0xcd, 0xab)
STORE_TEST(t_aligned_beu16, __builtin_stdc_store8_aligned_beu16, 0xABCD, 0xab, 0xcd)
STORE_TEST(t_aligned_leu32, __builtin_stdc_store8_aligned_leu32, 0x12345678u, 0x78, 0x56, 0x34, 0x12)
STORE_TEST(t_aligned_beu32, __builtin_stdc_store8_aligned_beu32, 0x12345678u, 0x12, 0x34, 0x56, 0x78)
STORE_TEST(t_aligned_leu64, __builtin_stdc_store8_aligned_leu64, 0x1122334455667788ull, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11)
STORE_TEST(t_aligned_beu64, __builtin_stdc_store8_aligned_beu64, 0x1122334455667788ull, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88)
STORE_TEST(t_aligned_les8, __builtin_stdc_store8_aligned_les8, -128, 0x80)
STORE_TEST(t_aligned_bes8, __builtin_stdc_store8_aligned_bes8, -128, 0x80)
STORE_TEST(t_aligned_les16, __builtin_stdc_store8_aligned_les16, -2, 0xfe, 0xff)
STORE_TEST(t_aligned_bes16, __builtin_stdc_store8_aligned_bes16, -2, 0xff, 0xfe)
STORE_TEST(t_aligned_les32, __builtin_stdc_store8_aligned_les32, -2, 0xfe, 0xff, 0xff, 0xff)
STORE_TEST(t_aligned_bes32, __builtin_stdc_store8_aligned_bes32, -2, 0xff, 0xff, 0xff, 0xfe)
STORE_TEST(t_aligned_les64, __builtin_stdc_store8_aligned_les64, (-9223372036854775807ll - 1), 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x80)
STORE_TEST(t_aligned_bes64, __builtin_stdc_store8_aligned_bes64, (-9223372036854775807ll - 1), 0x80, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0)

// Writes land at the given offset and leave neighbouring bytes alone.
constexpr bool test_offset() {
  unsigned char buf[6] = {1, 1, 1, 1, 1, 1};
  __builtin_stdc_store8_beu16(0xABCD, buf + 2);
  return buf[0] == 1 && buf[1] == 1 && buf[2] == 0xAB && buf[3] == 0xCD &&
         buf[4] == 1 && buf[5] == 1;
}
static_assert(test_offset(), "");

// Later stores overwrite earlier ones.
constexpr bool test_overwrite() {
  unsigned char buf[4] = {};
  __builtin_stdc_store8_leu32(0xFFFFFFFFu, buf);
  __builtin_stdc_store8_beu16(0x0102, buf + 1);
  return buf[0] == 0xFF && buf[1] == 0x01 && buf[2] == 0x02 && buf[3] == 0xFF;
}
static_assert(test_overwrite(), "");

// Storing initializes an otherwise uninitialized buffer.
constexpr bool test_initializes() {
  unsigned char buf[4];
  __builtin_stdc_store8_leu32(0x04030201u, buf);
  return buf[0] == 1 && buf[1] == 2 && buf[2] == 3 && buf[3] == 4;
}
static_assert(test_initializes(), "");

// An aligned store at an offset that is only as aligned as the width needs.
constexpr bool test_aligned_offset() {
  alignas(8) unsigned char buf[8] = {};
  __builtin_stdc_store8_aligned_leu16(0x0201, buf + 2);
  __builtin_stdc_store8_aligned_beu32(0x03040506u, buf + 4);
  return buf[2] == 1 && buf[3] == 2 && buf[4] == 3 && buf[7] == 6;
}
static_assert(test_aligned_offset(), "");

// Storing through a pointer to a larger object's byte range.
struct S { alignas(4) unsigned char bytes[4]; unsigned char tail; };
constexpr bool test_member() {
  S s = {};
  __builtin_stdc_store8_aligned_leu32(0xAABBCCDDu, s.bytes);
  return s.bytes[0] == 0xDD && s.bytes[3] == 0xAA && s.tail == 0;
}
static_assert(test_member(), "");

// Errors.
constexpr bool test_oob() { // both-error{{constexpr function never produces a constant expression}}
  unsigned char buf[2];
  __builtin_stdc_store8_leu32(1, buf); // both-note 2{{cannot refer to element 3 of array of 2 elements in a constant expression}}
  return true;
}
static_assert(test_oob(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_oob()'}}

constexpr bool test_oob_offset() { // both-error{{constexpr function never produces a constant expression}}
  unsigned char buf[4];
  __builtin_stdc_store8_leu32(1, buf + 1); // both-note 2{{cannot refer to element 4 of array of 4 elements in a constant expression}}
  return true;
}
static_assert(test_oob_offset(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_oob_offset()'}}

constexpr bool test_one_past_end() { // both-error{{constexpr function never produces a constant expression}}
  unsigned char buf[1];
  __builtin_stdc_store8_leu8(1, buf + 1); // both-note 2{{cannot refer to element 1 of array of 1 element in a constant expression}}
  return true;
}
static_assert(test_one_past_end(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_one_past_end()'}}

constexpr bool test_null() { // both-error{{constexpr function never produces a constant expression}}
  __builtin_stdc_store8_leu32(1, (unsigned char *)0); // both-warning{{null passed to a callee that requires a non-null argument}} both-note 2{{assignment to dereferenced null pointer is not allowed in a constant expression}}
  return true;
}
static_assert(test_null(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_null()'}}

constexpr unsigned char read_only[4] = {};
constexpr bool test_const_object() { // both-error{{constexpr function never produces a constant expression}}
  __builtin_stdc_store8_leu32(1, const_cast<unsigned char *>(read_only)); // both-note 2{{a constant expression cannot modify an object that is visible outside that expression}}
  return true;
}
static_assert(test_const_object(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_const_object()'}}

// aligned_* requires the pointer to be aligned to the width of the value.
constexpr bool test_misaligned16() { // both-error{{constexpr function never produces a constant expression}}
  alignas(8) unsigned char buf[9] = {};
  __builtin_stdc_store8_aligned_leu16(1, buf + 1); // both-note 2{{'__builtin_stdc_store8_aligned_leu16' requires a pointer aligned to 2 bytes, but the given pointer is only aligned to 1 byte}}
  return true;
}
static_assert(test_misaligned16(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_misaligned16()'}}

constexpr bool test_misaligned32() { // both-error{{constexpr function never produces a constant expression}}
  alignas(8) unsigned char buf[8] = {};
  __builtin_stdc_store8_aligned_beu32(1, buf + 2); // both-note 2{{'__builtin_stdc_store8_aligned_beu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 2 bytes}}
  return true;
}
static_assert(test_misaligned32(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_misaligned32()'}}

constexpr bool test_unaligned_buffer() { // both-error{{constexpr function never produces a constant expression}}
  unsigned char buf[8] = {};
  __builtin_stdc_store8_aligned_les64(1, buf); // both-note 2{{'__builtin_stdc_store8_aligned_les64' requires a pointer aligned to 8 bytes, but the given pointer is only aligned to 1 byte}}
  return true;
}
static_assert(test_unaligned_buffer(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_unaligned_buffer()'}}

// The functions return void.
static_assert(__is_same(decltype(__builtin_stdc_store8_leu8(0, (unsigned char *)0)), void), "");
static_assert(__is_same(decltype(__builtin_stdc_store8_aligned_bes64(0, (unsigned char *)0)), void), "");

// An unsigned variant converts a negative argument to the unsigned type.
constexpr bool test_negative_to_unsigned() {
  unsigned char buf[2] = {};
  __builtin_stdc_store8_leu16(-2, buf);
  return buf[0] == 0xFE && buf[1] == 0xFF;
}
static_assert(test_negative_to_unsigned(), "");

// The 8-bit aligned variants never need alignment.
constexpr bool test_aligned8_unaligned_buffer() {
  unsigned char buf[3] = {};
  __builtin_stdc_store8_aligned_leu8(0x11, buf + 1);
  __builtin_stdc_store8_aligned_bes8(-1, buf + 2);
  return buf[0] == 0 && buf[1] == 0x11 && buf[2] == 0xFF;
}
static_assert(test_aligned8_unaligned_buffer(), "");

// Scalar (non-array) destinations: an 8-bit store is fine, wider ones run past
// the object.
constexpr bool test_scalar_ok() {
  unsigned char c = 0;
  __builtin_stdc_store8_leu8(0x42, &c);
  __builtin_stdc_store8_aligned_beu8(0x43, &c);
  return c == 0x43;
}
static_assert(test_scalar_ok(), "");

constexpr bool test_scalar_oob() { // both-error{{constexpr function never produces a constant expression}}
  alignas(4) unsigned char c = 0;
  __builtin_stdc_store8_aligned_leu32(1, &c); // both-note 2{{cannot refer to element 3 of non-array object in a constant expression}}
  return true;
}
static_assert(test_scalar_oob(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_scalar_oob()'}}

// An aligned store one past the end is out of bounds, whether or not the
// pointer is aligned.
constexpr bool test_aligned_one_past_end() { // both-error{{constexpr function never produces a constant expression}}
  alignas(8) unsigned char buf[4] = {};
  __builtin_stdc_store8_aligned_leu32(1, buf + 4); // both-note 2{{cannot refer to element 7 of array of 4 elements in a constant expression}}
  return true;
}
static_assert(test_aligned_one_past_end(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_aligned_one_past_end()'}}

constexpr bool test_aligned_one_past_end_misaligned() { // both-error{{constexpr function never produces a constant expression}}
  alignas(8) unsigned char buf[5] = {};
  __builtin_stdc_store8_aligned_leu32(1, buf + 5); // both-note 2{{'__builtin_stdc_store8_aligned_leu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 1 byte}}
  return true;
}
static_assert(test_aligned_one_past_end_misaligned(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_aligned_one_past_end_misaligned()'}}

// The signed and big-endian aligned variants check alignment too.
constexpr bool test_misaligned_les32() { // both-error{{constexpr function never produces a constant expression}}
  alignas(8) unsigned char buf[9] = {};
  __builtin_stdc_store8_aligned_les32(1, buf + 1); // both-note 2{{'__builtin_stdc_store8_aligned_les32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 1 byte}}
  return true;
}
static_assert(test_misaligned_les32(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_misaligned_les32()'}}

constexpr bool test_misaligned_bes32() { // both-error{{constexpr function never produces a constant expression}}
  alignas(8) unsigned char buf[9] = {};
  __builtin_stdc_store8_aligned_bes32(1, buf + 1); // both-note 2{{'__builtin_stdc_store8_aligned_bes32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 1 byte}}
  return true;
}
static_assert(test_misaligned_bes32(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_misaligned_bes32()'}}

// Alignment of a pointer to a member depends on the member's offset.
struct FieldOffset { unsigned char pad; unsigned char data[8]; };
constexpr bool test_member_misaligned() { // both-error{{constexpr function never produces a constant expression}}
  alignas(8) FieldOffset s = {};
  __builtin_stdc_store8_aligned_leu32(1, s.data); // both-note 2{{'__builtin_stdc_store8_aligned_leu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 1 byte}}
  return true;
}
static_assert(test_member_misaligned(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_member_misaligned()'}}

constexpr bool test_array_member_misaligned() { // both-error{{constexpr function never produces a constant expression}}
  alignas(8) FieldOffset s[2] = {};
  __builtin_stdc_store8_aligned_leu32(1, s[1].data); // both-note 2{{'__builtin_stdc_store8_aligned_leu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 2 bytes}}
  return true;
}
static_assert(test_array_member_misaligned(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_array_member_misaligned()'}}

union FieldUnion { unsigned char c; unsigned char data[8]; };
constexpr bool test_union_member() {
  alignas(8) FieldUnion u = {.data = {}};
  __builtin_stdc_store8_aligned_leu32(0x12345678u, u.data);
  return u.data[0] == 0x78 && u.data[3] == 0x12;
}
static_assert(test_union_member(), "");

// Alignment from a typedef'd attribute applies to the offset arithmetic.
typedef unsigned char AlignedBuf8[8] __attribute__((aligned(8)));
constexpr bool test_typedef_aligned() {
  AlignedBuf8 buf = {};
  __builtin_stdc_store8_aligned_leu32(0x01020304u, buf + 4);
  return buf[4] == 4 && buf[7] == 1;
}
static_assert(test_typedef_aligned(), "");

constexpr bool test_typedef_offset_fail() { // both-error{{constexpr function never produces a constant expression}}
  AlignedBuf8 buf = {};
  __builtin_stdc_store8_aligned_leu32(1, buf + 2); // both-note 2{{'__builtin_stdc_store8_aligned_leu32' requires a pointer aligned to 4 bytes, but the given pointer is only aligned to 2 bytes}}
  return true;
}
static_assert(test_typedef_offset_fail(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_typedef_offset_fail()'}}

// Dynamically allocated storage is at least as aligned as the allocation
// function guarantees.
constexpr bool test_dynamic_alloc() {
  unsigned char *p = new unsigned char[8];
  __builtin_stdc_store8_aligned_beu64(0x0102030405060708ull, p);
  bool ok = p[0] == 1 && p[7] == 8;
  delete[] p;
  return ok;
}
static_assert(test_dynamic_alloc(), "");
