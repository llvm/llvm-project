// RUN: %clang_cc1 -ffreestanding -triple x86_64-apple-macosx10.8.0 -target-feature +sse4.1 -emit-llvm %s -o - | FileCheck %s

#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>

__m128 test_mm_add_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_add_ps
  // CHECK: fadd <4 x float>
  return _mm_add_ps(A, B);
}

__m128 test_mm_and_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_and_ps
  // CHECK: and <4 x i32> %{{.*}}, %{{.*}}
  return _mm_and_ps(A, B);
}

__m128 test_mm_div_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_div_ps
  // CHECK: fdiv <4 x float>
  return _mm_div_ps(A, B);
}

__m128 test_mm_mul_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_mul_ps
  // CHECK: fmul <4 x float>
  return _mm_mul_ps(A, B);
}

__m128 test_mm_or_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_or_ps
  // CHECK: or <4 x i32> %{{.*}}, %{{.*}}
  return _mm_or_ps(A, B);
}

__m128 test_mm_sub_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_sub_ps
  // CHECK: fsub <4 x float>
  return _mm_sub_ps(A, B);
}

__m128 test_mm_xor_ps(__m128 A, __m128 B) {
  // CHECK-LABEL: test_mm_xor_ps
  // CHECK: xor <4 x i32> %{{.*}}, %{{.*}}
  return _mm_xor_ps(A, B);
}

__m128 test_rsqrt_ss(__m128 x) {
  // CHECK: define {{.*}} @test_rsqrt_ss
  // CHECK: call <4 x float> @llvm.x86.sse.rsqrt.ss
  // CHECK: extractelement <4 x float> {{.*}}, i32 0
  // CHECK: extractelement <4 x float> {{.*}}, i32 1
  // CHECK: extractelement <4 x float> {{.*}}, i32 2
  // CHECK: extractelement <4 x float> {{.*}}, i32 3
  return _mm_rsqrt_ss(x);
}

__m128 test_rcp_ss(__m128 x) {
  // CHECK: define {{.*}} @test_rcp_ss
  // CHECK: call <4 x float> @llvm.x86.sse.rcp.ss
  // CHECK: extractelement <4 x float> {{.*}}, i32 0
  // CHECK: extractelement <4 x float> {{.*}}, i32 1
  // CHECK: extractelement <4 x float> {{.*}}, i32 2
  // CHECK: extractelement <4 x float> {{.*}}, i32 3
  return _mm_rcp_ss(x);
}

__m128 test_sqrt_ss(__m128 x) {
  // CHECK: define {{.*}} @test_sqrt_ss
  // CHECK: call <4 x float> @llvm.x86.sse.sqrt.ss
  // CHECK: extractelement <4 x float> {{.*}}, i32 0
  // CHECK: extractelement <4 x float> {{.*}}, i32 1
  // CHECK: extractelement <4 x float> {{.*}}, i32 2
  // CHECK: extractelement <4 x float> {{.*}}, i32 3
  return _mm_sqrt_ss(x);
}

__m128 test_loadl_pi(__m128 x, void* y) {
  // CHECK: define {{.*}} @test_loadl_pi
  // CHECK: load <2 x float>, <2 x float>* {{.*}}, align 1{{$}}
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 0, i32 1
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  return _mm_loadl_pi(x,y);
}

__m128 test_loadh_pi(__m128 x, void* y) {
  // CHECK: define {{.*}} @test_loadh_pi
  // CHECK: load <2 x float>, <2 x float>* {{.*}}, align 1{{$}}
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 0, i32 1
  // CHECK: shufflevector {{.*}} <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  return _mm_loadh_pi(x,y);
}

__m128 test_load_ss(void* y) {
  // CHECK: define {{.*}} @test_load_ss
  // CHECK: load float, float* {{.*}}, align 1{{$}}
  return _mm_load_ss(y);
}

__m128 test_load1_ps(void* y) {
  // CHECK: define {{.*}} @test_load1_ps
  // CHECK: load float, float* {{.*}}, align 1{{$}}
  return _mm_load1_ps(y);
}

void test_store_ss(__m128 x, void* y) {
  // CHECK-LABEL: define void @test_store_ss
  // CHECK: store {{.*}} float* {{.*}}, align 1{{$}}
  _mm_store_ss(y, x);
}

__m128 test_mm_cmpeq_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpeq_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 0)
  return _mm_cmpeq_ss(__a, __b);
}

__m128 test_mm_cmplt_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmplt_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 1)
  return _mm_cmplt_ss(__a, __b);
}

__m128 test_mm_cmple_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmple_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_cmple_ss(__a, __b);
}

__m128 test_mm_cmpunord_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpunord_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 3)
  return _mm_cmpunord_ss(__a, __b);
}

__m128 test_mm_cmpneq_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpneq_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 4)
  return _mm_cmpneq_ss(__a, __b);
}

__m128 test_mm_cmpnlt_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnlt_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 5)
  return _mm_cmpnlt_ss(__a, __b);
}

__m128 test_mm_cmpnle_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnle_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 6)
  return _mm_cmpnle_ss(__a, __b);
}

__m128 test_mm_cmpord_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpord_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 7)
  return _mm_cmpord_ss(__a, __b);
}

__m128 test_mm_cmpgt_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpgt_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 1)
  return _mm_cmpgt_ss(__a, __b);
}

__m128 test_mm_cmpge_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpge_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_cmpge_ss(__a, __b);
}

__m128 test_mm_cmpngt_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpngt_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 5)
  return _mm_cmpngt_ss(__a, __b);
}

__m128 test_mm_cmpnge_ss(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnge_ss
  // CHECK: @llvm.x86.sse.cmp.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 6)
  return _mm_cmpnge_ss(__a, __b);
}

__m128 test_mm_cmpeq_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpeq_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 0)
  return _mm_cmpeq_ps(__a, __b);
}

__m128 test_mm_cmplt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmplt_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 1)
  return _mm_cmplt_ps(__a, __b);
}

__m128 test_mm_cmple_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmple_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_cmple_ps(__a, __b);
}

__m128 test_mm_cmpunord_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpunord_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 3)
  return _mm_cmpunord_ps(__a, __b);
}

__m128 test_mm_cmpneq_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpneq_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 4)
  return _mm_cmpneq_ps(__a, __b);
}

__m128 test_mm_cmpnlt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnlt_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 5)
  return _mm_cmpnlt_ps(__a, __b);
}

__m128 test_mm_cmpnle_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnle_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 6)
  return _mm_cmpnle_ps(__a, __b);
}

__m128 test_mm_cmpord_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpord_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 7)
  return _mm_cmpord_ps(__a, __b);
}

__m128 test_mm_cmpgt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpgt_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 1)
  return _mm_cmpgt_ps(__a, __b);
}

__m128 test_mm_cmpge_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpge_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 2)
  return _mm_cmpge_ps(__a, __b);
}

__m128 test_mm_cmpngt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpngt_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 5)
  return _mm_cmpngt_ps(__a, __b);
}

__m128 test_mm_cmpnge_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnge_ps
  // CHECK: @llvm.x86.sse.cmp.ps(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i8 6)
  return _mm_cmpnge_ps(__a, __b);
}

__m128 test_mm_undefined_ps() {
  // CHECK-LABEL: @test_mm_undefined_ps
  // CHECK: ret <4 x float> undef
  return _mm_undefined_ps();
}

