// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse -emit-llvm -ffp-exception-behavior=strict -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

__m128 test_mm_cmpeq_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpeq_ps
  // CHECK:         [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpeq_ps(__a, __b);
}

__m128 test_mm_cmpge_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpge_ps
  // CHECK:         [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpge_ps(__a, __b);
}

__m128 test_mm_cmpgt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpgt_ps
  // CHECK:         [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpgt_ps(__a, __b);
}

__m128 test_mm_cmple_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmple_ps
  // CHECK:         [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmple_ps(__a, __b);
}

__m128 test_mm_cmplt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmplt_ps
  // CHECK:         [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmplt_ps(__a, __b);
}

__m128 test_mm_cmpneq_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpneq_ps
  // CHECK:         [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpneq_ps(__a, __b);
}

__m128 test_mm_cmpnge_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnge_ps
  // CHECK:         [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpnge_ps(__a, __b);
}

__m128 test_mm_cmpngt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpngt_ps
  // CHECK:         [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpngt_ps(__a, __b);
}

__m128 test_mm_cmpnle_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnle_ps
  // CHECK:         [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpnle_ps(__a, __b);
}

__m128 test_mm_cmpnlt_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpnlt_ps
  // CHECK:         [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpnlt_ps(__a, __b);
}

__m128 test_mm_cmpord_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpord_ps
  // CHECK:         [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpord_ps(__a, __b);
}

__m128 test_mm_cmpunord_ps(__m128 __a, __m128 __b) {
  // CHECK-LABEL: @test_mm_cmpunord_ps
  // CHECK:         [[CMP:%.*]] = call <4 x i1> @llvm.experimental.constrained.fcmp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <4 x i32> [[SEXT]] to <4 x float>
  // CHECK-NEXT:    ret <4 x float> [[BC]]
  return _mm_cmpunord_ps(__a, __b);
}
