// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx10.1 -emit-llvm -o - -Werror -pedantic | FileCheck %s

// Tests ACE v1 BSR (Block Scale Register) operations emit correct LLVM IR

#include <immintrin.h>
#include <stdint.h>

void test_bsr0_init(void) {
  // CHECK-LABEL: @test_bsr0_init
  // CHECK: call void @llvm.x86.acev1.bsr0init()
  _bsr0_init();
}

void test_bsr0_insertfull(__m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_bsr0_insertfull
  // CHECK: call void @llvm.x86.acev1.bsr0movf(<64 x i8> %{{.*}}, <64 x i8> %{{.*}})
  _bsr0_insertfull(src1, src2);
}

void test_bsr0_inserth(__m512i src) {
  // CHECK-LABEL: @test_bsr0_inserth
  // CHECK: call void @llvm.x86.acev1.bsr0movhinsert(<64 x i8> %{{.*}})
  _bsr0_inserth(src);
}

__m512i test_bsr0_extracth(void) {
  // CHECK-LABEL: @test_bsr0_extracth
  // CHECK: call <64 x i8> @llvm.x86.acev1.bsr0movhextract()
  return _bsr0_extracth();
}

void test_bsr0_insertl(__m512i src) {
  // CHECK-LABEL: @test_bsr0_insertl
  // CHECK: call void @llvm.x86.acev1.bsr0movlinsert(<64 x i8> %{{.*}})
  _bsr0_insertl(src);
}

__m512i test_bsr0_extractl(void) {
  // CHECK-LABEL: @test_bsr0_extractl
  // CHECK: call <64 x i8> @llvm.x86.acev1.bsr0movlextract()
  return _bsr0_extractl();
}

// Test BSR operations sequence
void test_bsr_sequence(void) {
  // CHECK-LABEL: @test_bsr_sequence
  // CHECK: call void @llvm.x86.acev1.bsr0init()
  // CHECK: call void @llvm.x86.acev1.bsr0movf
  // CHECK: call <64 x i8> @llvm.x86.acev1.bsr0movhextract()
  // CHECK: call <64 x i8> @llvm.x86.acev1.bsr0movlextract()
  _bsr0_init();

  __m512i zmm1 = _mm512_set1_epi32(1);
  __m512i zmm2 = _mm512_set1_epi32(2);
  _bsr0_insertfull(zmm1, zmm2);

  __m512i high = _bsr0_extracth();
  __m512i low = _bsr0_extractl();

  (void)high;
  (void)low;
}
