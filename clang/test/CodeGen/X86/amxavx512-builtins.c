// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +amx-tile -target-feature +amx-avx512 \
// RUN: -target-feature +avx10.2 -emit-llvm -o - -Wall -Werror -pedantic -Wno-gnu-statement-expression -flax-vector-conversions=none | FileCheck %s

#include <immintrin.h>
#include <stddef.h>

__m512 test_tile_cvtrowd2ps(unsigned int A) {
  // CHECK-LABEL: @test_tile_cvtrowd2ps(
  // CHECK: call <16 x float> @llvm.x86.tcvtrowd2ps(i8 1, i32 %{{.*}})
  return _tile_cvtrowd2ps(1, A);
}

__m512bh test_tile_cvtrowps2bf16h(unsigned int A) {
  // CHECK-LABEL: @test_tile_cvtrowps2bf16h(
  // CHECK: call <32 x bfloat> @llvm.x86.tcvtrowps2bf16h(i8 1, i32 %{{.*}})
  return _tile_cvtrowps2bf16h(1, A);
}

__m512bh test_tile_cvtrowps2bf16l(unsigned int A) {
  // CHECK-LABEL: @test_tile_cvtrowps2bf16l(
  // CHECK: call <32 x bfloat> @llvm.x86.tcvtrowps2bf16l(i8 1, i32 %{{.*}})
  return _tile_cvtrowps2bf16l(1, A);
}

__m512h test_tile_cvtrowps2phh(unsigned int A) {
  // CHECK-LABEL: @test_tile_cvtrowps2phh(
  // CHECK: call <32 x half> @llvm.x86.tcvtrowps2phh(i8 1, i32 %{{.*}})
  return _tile_cvtrowps2phh(1, A);
}

__m512h test_tile_cvtrowps2phl(unsigned int A) {
  // CHECK-LABEL: @test_tile_cvtrowps2phl(
  // CHECK: call <32 x half> @llvm.x86.tcvtrowps2phl(i8 1, i32 %{{.*}})
  return _tile_cvtrowps2phl(1, A);
}

__m512i test_tile_movrow(unsigned int A) {
  // CHECK-LABEL: @test_tile_movrow
  // CHECK: %1 = call <16 x i32> @llvm.x86.tilemovrow(i8 1, i32 %{{.*}})
  return _tile_movrow(1, A);
}

__m512 test_tile_cvtrowd2psi(void) {
  // CHECK-LABEL: @test_tile_cvtrowd2psi(
  // CHECK: call <16 x float> @llvm.x86.tcvtrowd2psi(i8 1, i32 2)
  return _tile_cvtrowd2psi(1, 2);
}

__m512bh test_tile_cvtrowps2bf16hi(void) {
  // CHECK-LABEL: @test_tile_cvtrowps2bf16hi(
  // CHECK: call <32 x bfloat> @llvm.x86.tcvtrowps2bf16hi(i8 1, i32 2)
  return _tile_cvtrowps2bf16hi(1, 2);
}

__m512bh test_tile_cvtrowps2bf16li(void) {
  // CHECK-LABEL: @test_tile_cvtrowps2bf16li(
  // CHECK: call <32 x bfloat> @llvm.x86.tcvtrowps2bf16li(i8 1, i32 2)
  return _tile_cvtrowps2bf16li(1, 2);
}

__m512h test_tile_cvtrowps2phhi(void) {
  // CHECK-LABEL: @test_tile_cvtrowps2phhi(
  // CHECK: call <32 x half> @llvm.x86.tcvtrowps2phhi(i8 1, i32 2)
  return _tile_cvtrowps2phhi(1, 2);
}

__m512h test_tile_cvtrowps2phli(void) {
  // CHECK-LABEL: @test_tile_cvtrowps2phli(
  // CHECK: call <32 x half> @llvm.x86.tcvtrowps2phli(i8 1, i32 2)
  return _tile_cvtrowps2phli(1, 2);
}

__m512i test_tile_movrowi(void) {
  // CHECK-LABEL: @test_tile_movrowi
  // CHECK: %0 = call <16 x i32> @llvm.x86.tilemovrowi(i8 1, i32 2)
  return _tile_movrowi(1, 2);
}
