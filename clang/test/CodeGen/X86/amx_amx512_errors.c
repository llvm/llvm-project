// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +amx-avx512 -target-feature +avx10.2 -verify

#include <immintrin.h>
#include <stddef.h>

void test_tile_mmultf32ps() {
  _tile_cvtrowd2psi(16, 2); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_cvtrowd2psi(1, 260); // expected-error {{argument value 260 is outside the valid range [0, 255]}}
}

