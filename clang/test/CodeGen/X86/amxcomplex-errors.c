// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +amx-complex -emit-llvm -fsyntax-only -verify

#include <immintrin.h>
#include <stddef.h>
void test_tile_cmmimfp16ps() {
  _tile_cmmimfp16ps(16, 2, 3); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_cmmimfp16ps(1, 26, 3); // expected-error {{argument value 26 is outside the valid range [0, 7]}}
  _tile_cmmimfp16ps(1, 2, 36); // expected-error {{argument value 36 is outside the valid range [0, 7]}}
  _tile_cmmimfp16ps(1, 1, 3);  // expected-error {{tile arguments must refer to different tiles}}
}

void test_tile_cmmrlfp16ps() {
  _tile_cmmrlfp16ps(16, 2, 3); // expected-error {{argument value 16 is outside the valid range [0, 7]}}
  _tile_cmmrlfp16ps(1, 26, 3); // expected-error {{argument value 26 is outside the valid range [0, 7]}}
  _tile_cmmrlfp16ps(1, 2, 36); // expected-error {{argument value 36 is outside the valid range [0, 7]}}
  _tile_cmmrlfp16ps(1, 1, 3);  // expected-error {{tile arguments must refer to different tiles}}
}
