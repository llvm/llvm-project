// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +acev1 \
// RUN: -target-feature +avx512f -emit-llvm -o - -Werror -pedantic | FileCheck %s

// Tests ACE v1 BSR (Block Sparse Register) operations emit correct LLVM IR

#include <immintrin.h>
#include <stdint.h>

void test_bsr_init(void) {
  // CHECK-LABEL: @test_bsr_init
  // CHECK: call void @llvm.x86.bsrinit()
  _bsr0_init();
}

void test_bsr_movf(__m512i src1, __m512i src2) {
  // CHECK-LABEL: @test_bsr_movf
  // CHECK: call void @llvm.x86.bsrmovf(<16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  _bsr0_movf(src1, src2);
}

void test_bsr_movh_set(__m512i src) {
  // CHECK-LABEL: @test_bsr_movh_set
  // CHECK: call void @llvm.x86.bsrmovh.set(<16 x i32> %{{.*}})
  _bsr0_movh_set(src);
}

__m512i test_bsr_movh_get(void) {
  // CHECK-LABEL: @test_bsr_movh_get
  // CHECK: call <16 x i32> @llvm.x86.bsrmovh.get()
  return _bsr0_movh_get();
}

void test_bsr_movl_set(__m512i src) {
  // CHECK-LABEL: @test_bsr_movl_set
  // CHECK: call void @llvm.x86.bsrmovl.set(<16 x i32> %{{.*}})
  _bsr0_movl_set(src);
}

__m512i test_bsr_movl_get(void) {
  // CHECK-LABEL: @test_bsr_movl_get
  // CHECK: call <16 x i32> @llvm.x86.bsrmovl.get()
  return _bsr0_movl_get();
}

// Test BSR operations sequence
void test_bsr_sequence(void) {
  // CHECK-LABEL: @test_bsr_sequence
  // CHECK: call void @llvm.x86.bsrinit()
  // CHECK: call void @llvm.x86.bsrmovf
  // CHECK: call <16 x i32> @llvm.x86.bsrmovh.get()
  // CHECK: call <16 x i32> @llvm.x86.bsrmovl.get()
  _bsr0_init();

  __m512i zmm1 = _mm512_set1_epi32(1);
  __m512i zmm2 = _mm512_set1_epi32(2);
  _bsr0_movf(zmm1, zmm2);

  __m512i high = _bsr0_movh_get();
  __m512i low = _bsr0_movl_get();

  (void)high;
  (void)low;
}
