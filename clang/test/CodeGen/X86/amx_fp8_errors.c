// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +amx-tile -target-feature +amx-fp8 -verify

#include <immintrin.h>

void test_amx(void *data) {
  _tile_dpbf8ps(4, 3, 3); // expected-error {{tile arguments must refer to different tiles}}
  _tile_dpbhf8ps(4, 3, 3); // expected-error {{tile arguments must refer to different tiles}}
  _tile_dphbf8ps(4, 3, 3); // expected-error {{tile arguments must refer to different tiles}}
  _tile_dphf8ps(4, 3, 3); // expected-error {{tile arguments must refer to different tiles}}
}
