// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +amx-int8 -target-feature +amx-transpose -target-feature +amx-movrs \
// RUN: -verify

#include <immintrin.h>
#include <stddef.h>

void test_tile_2rpntlvwz0rs(const void *A, size_t B) {
  _tile_2rpntlvwz0rs(8, A, B); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_tile_2rpntlvwz0rst1(const void *A, size_t B) {
  _tile_2rpntlvwz0rst1(8, A, B); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_tile_2rpntlvwz1rs(const void *A, size_t B) {
  _tile_2rpntlvwz1rs(8, A, B); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

void test_tile_2rpntlvwz1rst1(const void *A, size_t B) {
  _tile_2rpntlvwz1rst1(8, A, B); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}
