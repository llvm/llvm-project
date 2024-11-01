// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +amx-int8 -target-feature +amx-bf16 -target-feature +amx-transpose \
// RUN: -target-feature +avx512f -target-feature +amx-element-evex -verify

#include <immintrin.h>
#include <stddef.h>
#include <immintrin.h>
#include <stddef.h>

// Transpose
void test_tile_2rpntlvwz0(const void *A, size_t B) {
  _tile_2rpntlvwz0(8, A, B); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_tile_2rpntlvwz0t1(const void *A, size_t B) {
  _tile_2rpntlvwz0t1(8, A, B); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_tile_2rpntlvwz1(const void *A, size_t B) {
  _tile_2rpntlvwz1(8, A, B); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_tile_2rpntlvwz1t1(const void *A, size_t B) {
  _tile_2rpntlvwz1t1(8, A, B); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_tile_transposed()
{
  _tile_transposed(8, 2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_transposed(1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}
