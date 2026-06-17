// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-linux-gnu
// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-linux-gnu -fexperimental-new-constant-interpreter

// Test that offsetof correctly zero-extends unsigned array indices >= 128.
// Previously, Clang would sign-extend uint8_t indices >= 128, producing
// a large bogus offset value instead of the correct one.
// Also tests that negative indices and oversized __uint128_t indices are rejected.
// https://github.com/llvm/llvm-project/issues/199319

#include <stdint.h>
#include <stddef.h>

struct MyStruct {
    void *ptrs[256];
};

// Unsigned indices that were previously sign-extended must be zero-extended.
_Static_assert(__builtin_offsetof(struct MyStruct, ptrs[(uint8_t)127]) == 127 * sizeof(void *),
               "offsetof with uint8_t index 127 should be correct");

_Static_assert(__builtin_offsetof(struct MyStruct, ptrs[(uint8_t)128]) == 128 * sizeof(void *),
               "offsetof with uint8_t index 128 should be correctly zero-extended, not sign-extended");

_Static_assert(__builtin_offsetof(struct MyStruct, ptrs[(uint8_t)255]) == 255 * sizeof(void *),
               "offsetof with uint8_t index 255 should be correctly zero-extended, not sign-extended");

// Negative indices must be rejected.
struct NegIdxStruct { int a; int x[1]; };
_Static_assert(__builtin_offsetof(struct NegIdxStruct, x[-1]) == 0, ""); // expected-error {{not an integral constant expression}} expected-note {{subexpression not valid in a constant expression}}

// __uint128_t indices >= 0x8000000000000000 must be rejected.
_Static_assert(__builtin_offsetof(struct NegIdxStruct, x[(__uint128_t)0x8000000000000000]) == 0, ""); // expected-error {{not an integral constant expression}} expected-note {{subexpression not valid in a constant expression}}

// __uint128_t indices > UINT64_MAX must be rejected (e.g. adding another zero:
// old code would truncate 2^64 to 0 via PT_Uint64 cast, silently producing a
// wrong result instead of an error).
_Static_assert(__builtin_offsetof(struct NegIdxStruct, x[((__uint128_t)1 << 64)]) == 0, ""); // expected-error {{not an integral constant expression}} expected-note {{subexpression not valid in a constant expression}}

// A uint64_t index that causes index*sizeof(element) to overflow int64_t must
// be rejected.  4611686018427387904 * sizeof(short)==2 == 2^63 > INT64_MAX.
struct ShortArray { short data[2]; };
_Static_assert(__builtin_offsetof(struct ShortArray, data[(uint64_t)4611686018427387904ULL]) == 0, ""); // expected-error {{not an integral constant expression}} expected-note {{subexpression not valid in a constant expression}}
