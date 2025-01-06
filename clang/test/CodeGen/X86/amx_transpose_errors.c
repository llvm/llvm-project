// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +amx-int8 -target-feature +amx-bf16 -target-feature +amx-transpose \
// RUN: -target-feature +avx512f -target-feature +amx-fp16 -target-feature +amx-complex -verify

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

void test_tile_tdpbf16ps()
{
  _tile_tdpbf16ps(8, 2, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_tdpbf16ps(1, 8, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_tdpbf16ps(1, 2, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_tdpbf16ps(1, 1, 3);  // expected-error {{tile arguments must refer to different tiles}}
  _tile_tdpbf16ps(1, 2, 1);  // expected-error {{tile arguments must refer to different tiles}}
  _tile_tdpbf16ps(1, 2, 2);  // expected-error {{tile arguments must refer to different tiles}}
}

void test_tile_tdpfp16ps()
{
  _tile_tdpfp16ps(8, 5, 6); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_tdpfp16ps(1, 8, 6); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_tdpfp16ps(1, 5, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_tdpfp16ps(1, 1, 3);  // expected-error {{tile arguments must refer to different tiles}}
  _tile_tdpfp16ps(1, 2, 1);  // expected-error {{tile arguments must refer to different tiles}}
  _tile_tdpfp16ps(1, 2, 2);  // expected-error {{tile arguments must refer to different tiles}}
}

void test_tile_transposed()
{
  _tile_transposed(8, 2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  _tile_transposed(1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_tile_tcmmimfp16ps() {
  _tile_tcmmimfp16ps(16, 2, 3); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_tcmmimfp16ps(1, 26, 3); // expected-error {{argument value 26 is outside the valid range [0, 7]}}
  _tile_tcmmimfp16ps(1, 2, 36); // expected-error {{argument value 36 is outside the valid range [0, 7]}}
  _tile_tcmmimfp16ps(1, 1, 3);  // expected-error {{tile arguments must refer to different tiles}}
}

void test_tile_tcmmrlfp16ps() {
  _tile_tcmmrlfp16ps(16, 2, 3); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_tcmmrlfp16ps(1, 26, 3); // expected-error {{argument value 26 is outside the valid range [0, 7]}}
  _tile_tcmmrlfp16ps(1, 2, 36); // expected-error {{argument value 36 is outside the valid range [0, 7]}}
  _tile_tcmmrlfp16ps(1, 1, 3);  // expected-error {{tile arguments must refer to different tiles}}
}

void test_tile_conjtcmmimfp16ps() {
  _tile_conjtcmmimfp16ps(16, 2, 3); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_conjtcmmimfp16ps(1, 26, 3); // expected-error {{argument value 26 is outside the valid range [0, 7]}}
  _tile_conjtcmmimfp16ps(1, 2, 36); // expected-error {{argument value 36 is outside the valid range [0, 7]}}
  _tile_conjtcmmimfp16ps(1, 2, 1);  // expected-error {{tile arguments must refer to different tiles}}
}

void test_tile_conjtfp16() {
  _tile_conjtfp16(16, 2); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_conjtfp16(1, 26); // expected-error {{argument value 26 is outside the valid range [0, 7]}}
}
