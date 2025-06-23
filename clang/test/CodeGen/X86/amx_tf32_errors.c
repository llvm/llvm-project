// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +amx-tf32 -target-feature +amx-transpose -verify

#include <immintrin.h>
#include <stddef.h>

void test_tile_mmultf32ps() {
  _tile_mmultf32ps(16, 2, 3); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_mmultf32ps(1, 26, 3); // expected-error {{argument value 26 is outside the valid range [0, 7]}}
  _tile_mmultf32ps(1, 2, 36); // expected-error {{argument value 36 is outside the valid range [0, 7]}}
  _tile_mmultf32ps(1, 1, 3);  // expected-error {{tile arguments must refer to different tiles}}
  _tile_mmultf32ps(1, 2, 1);  // expected-error {{tile arguments must refer to different tiles}}
  _tile_mmultf32ps(1, 3, 3);  // expected-error {{tile arguments must refer to different tiles}}
}

void test_tile_tmmultf32ps() {
  _tile_tmmultf32ps(16, 2, 3); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_tmmultf32ps(1, 26, 3); // expected-error {{argument value 26 is outside the valid range [0, 7]}}
  _tile_tmmultf32ps(1, 2, 36); // expected-error {{argument value 36 is outside the valid range [0, 7]}}
  _tile_tmmultf32ps(1, 1, 3);  // expected-error {{tile arguments must refer to different tiles}}
  _tile_tmmultf32ps(1, 2, 1);  // expected-error {{tile arguments must refer to different tiles}}
  _tile_tmmultf32ps(1, 2, 2);  // expected-error {{tile arguments must refer to different tiles}}
}
