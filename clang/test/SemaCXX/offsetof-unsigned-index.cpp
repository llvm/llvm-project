// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-linux-gnu -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-linux-gnu -std=c++11 -fexperimental-new-constant-interpreter

// Test that offsetof correctly zero-extends unsigned array indices >= 128.
// Previously, Clang would sign-extend uint8_t/uint16_t indices whose high bit
// was set, producing a large bogus offset value instead of the correct one.
// Also tests that negative indices and oversized __uint128_t indices are rejected.
// https://github.com/llvm/llvm-project/issues/199319

#include <cstdint>
#include <cstddef>

struct MyStruct {
    void *ptrs[256];
};

// uint8_t index: values >= 128 were incorrectly sign-extended
static_assert(__builtin_offsetof(MyStruct, ptrs[(uint8_t)127]) == 127 * sizeof(void *),
              "offsetof with uint8_t index 127 should be correct");
static_assert(__builtin_offsetof(MyStruct, ptrs[(uint8_t)128]) == 128 * sizeof(void *),
              "offsetof with uint8_t index 128 should be correctly zero-extended");
static_assert(__builtin_offsetof(MyStruct, ptrs[(uint8_t)255]) == 255 * sizeof(void *),
              "offsetof with uint8_t index 255 should be correctly zero-extended");

// uint16_t index: values >= 32768 were also affected
struct BigStruct {
    char data[65536];
};
static_assert(__builtin_offsetof(BigStruct, data[(uint16_t)32768]) == 32768,
              "offsetof with uint16_t index 32768 should be correctly zero-extended");

// Negative indices must be rejected.
struct NegIdxStruct { int a; int x[1]; };
static_assert(__builtin_offsetof(NegIdxStruct, x[-1]) == 0, ""); // expected-error {{not an integral constant expression}} expected-note {{subexpression not valid in a constant expression}}

// __uint128_t indices >= 0x8000000000000000 must be rejected.
static_assert(__builtin_offsetof(NegIdxStruct, x[(__uint128_t)0x8000000000000000]) == 0, ""); // expected-error {{not an integral constant expression}} expected-note {{subexpression not valid in a constant expression}}

// __uint128_t indices > UINT64_MAX must be rejected (e.g. adding another zero:
// old code would truncate 2^64 to 0 via PT_Uint64 cast, silently producing a
// wrong result instead of an error).
static_assert(__builtin_offsetof(NegIdxStruct, x[((__uint128_t)1 << 64)]) == 0, ""); // expected-error {{not an integral constant expression}} expected-note {{subexpression not valid in a constant expression}}

// A uint64_t index that causes index*sizeof(element) to overflow int64_t must
// be rejected.  4611686018427387904 * sizeof(short)==2 == 2^63 > INT64_MAX.
struct ShortArray { short data[2]; };
static_assert(__builtin_offsetof(ShortArray, data[(uint64_t)4611686018427387904ULL]) == 0, ""); // expected-error {{not an integral constant expression}} expected-note {{subexpression not valid in a constant expression}}
