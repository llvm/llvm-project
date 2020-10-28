// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse2 -emit-llvm -ffp-exception-behavior=strict -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

__m128d test_mm_cmpeq_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpeq_pd
  // CHECK:         [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"oeq", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK-NEXT:    ret <2 x double> [[BC]]
  return _mm_cmpeq_pd(A, B);
}

__m128d test_mm_cmpge_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpge_pd
  // CHECK:         [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK-NEXT:    ret <2 x double> [[BC]]
  return _mm_cmpge_pd(A, B);
}

__m128d test_mm_cmpgt_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpgt_pd
  // CHECK:         [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK-NEXT:    ret <2 x double> [[BC]]
  return _mm_cmpgt_pd(A, B);
}

__m128d test_mm_cmple_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmple_pd
  // CHECK:         [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ole", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK-NEXT:    ret <2 x double> [[BC]]
  return _mm_cmple_pd(A, B);
}

__m128d test_mm_cmplt_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmplt_pd
  // CHECK:         [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"olt", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK-NEXT:    ret <2 x double> [[BC]]
  return _mm_cmplt_pd(A, B);
}

__m128d test_mm_cmpneq_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpneq_pd
  // CHECK:         [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"une", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK-NEXT:    ret <2 x double> [[BC]]
  return _mm_cmpneq_pd(A, B);
}

__m128d test_mm_cmpnge_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpnge_pd
  // CHECK:         [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK-NEXT:    ret <2 x double> [[BC]]
  return _mm_cmpnge_pd(A, B);
}

__m128d test_mm_cmpngt_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpngt_pd
  // CHECK:         [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK-NEXT:    ret <2 x double> [[BC]]
  return _mm_cmpngt_pd(A, B);
}

__m128d test_mm_cmpnle_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpnle_pd
  // CHECK:         [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ugt", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK-NEXT:    ret <2 x double> [[BC]]
  return _mm_cmpnle_pd(A, B);
}

__m128d test_mm_cmpnlt_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpnlt_pd
  // CHECK:         [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uge", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK-NEXT:    ret <2 x double> [[BC]]
  return _mm_cmpnlt_pd(A, B);
}

__m128d test_mm_cmpord_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpord_pd
  // CHECK:         [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"ord", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK-NEXT:    ret <2 x double> [[BC]]
  return _mm_cmpord_pd(A, B);
}

__m128d test_mm_cmpunord_pd(__m128d A, __m128d B) {
  // CHECK-LABEL: test_mm_cmpunord_pd
  // CHECK:         [[CMP:%.*]] = call <2 x i1> @llvm.experimental.constrained.fcmp.v2f64(<2 x double> %{{.*}}, <2 x double> %{{.*}}, metadata !"uno", metadata !"fpexcept.strict")
  // CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
  // CHECK-NEXT:    [[BC:%.*]] = bitcast <2 x i64> [[SEXT]] to <2 x double>
  // CHECK-NEXT:    ret <2 x double> [[BC]]
  return _mm_cmpunord_pd(A, B);
}
