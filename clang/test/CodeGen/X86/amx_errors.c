// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +amx-tile   \
// RUN: -target-feature +amx-int8 -target-feature +amx-bf16 -target-feature +amx-fp16 -emit-llvm -fsyntax-only -verify

#include <immintrin.h>

void test_amx(void *data) {
  _tile_zero(16); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_loadd(19, data, 16); // expected-error {{argument value 19 is outside the valid range [0, 7]}}
  _tile_stream_loadd(23, data, 1); // expected-error {{argument value 23 is outside the valid range [0, 7]}}
  _tile_stored(88, data, 1); // expected-error {{argument value 88 is outside the valid range [0, 7]}}
  _tile_dpbssd(16, 2, 3); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_dpbssd(0, 16, 3); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_dpbuud(0, 2, 16); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_dpbsud(1, 1, 3); // expected-error {{tile arguments must refer to different tiles}}
  _tile_dpbsud(7, 1, 7); // expected-error {{tile arguments must refer to different tiles}}
  _tile_dpbsud(4, 3, 3); // expected-error {{tile arguments must refer to different tiles}}
  _tile_dpbf16ps(4, 3, 3); // expected-error {{tile arguments must refer to different tiles}}
  _tile_dpfp16ps(1, 1, 3); // expected-error {{tile arguments must refer to different tiles}}
  _tile_dpfp16ps(1, 2, 1); // expected-error {{tile arguments must refer to different tiles}}
  _tile_dpfp16ps(1, 2, 2); // expected-error {{tile arguments must refer to different tiles}}
}
