// RUN: %clang_cc1 -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefixes=OGCG --input-file=%t.ll %s

#include <immintrin.h>

__m128d test_mm_round_pd_roundeven(__m128d x) {
  // CIR-LABEL: test_mm_round_pd_roundeven
  // CIR: cir.roundeven %{{.*}} : !cir.vector<2 x !cir.double> {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}

  // LLVM-LABEL: test_mm_round_pd_roundeven
  // LLVM: call <2 x double> @llvm.experimental.constrained.roundeven.v2f64(<2 x double> %{{.*}}, metadata !"fpexcept.ignore")

  // OGCG-LABEL: test_mm_round_pd_roundeven
  // OGCG: call <2 x double> @llvm.experimental.constrained.roundeven.v2f64(<2 x double> %{{.*}}, metadata !"fpexcept.ignore")
  return _mm_round_pd(x, 0b1000);
}

__m128d test_mm_round_pd_mxcsr(__m128d x) {
  // CIR-LABEL: test_mm_round_pd_mxcsr
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.pd" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !s32i) -> !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_round_pd_mxcsr
  // LLVM: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 12)

  // OGCG-LABEL: test_mm_round_pd_mxcsr
  // OGCG: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 12)
  return _mm_round_pd(x, 0b1100);
}

__m128d test_mm_round_pd_fround_no_exc(__m128d x) {
  // CIR-LABEL: test_mm_round_pd_fround_no_exc
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.pd" %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !s32i) -> !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_round_pd_fround_no_exc
  // LLVM: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 0)

  // OGCG-LABEL: test_mm_round_pd_fround_no_exc
  // OGCG: call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %{{.*}}, i32 0)
  return _mm_round_pd(x, 0b0000);
}

__m128 test_mm_round_ps_floor(__m128 x) {
  // CIR-LABEL: test_mm_round_ps_floor
  // CIR: cir.floor %{{.*}} : !cir.vector<4 x !cir.float> {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}

  // LLVM-LABEL: test_mm_round_ps_floor
  // LLVM: call <4 x float> @llvm.experimental.constrained.floor.v4f32(<4 x float> %{{.*}}, metadata !"fpexcept.ignore")

  // OGCG-LABEL: test_mm_round_ps_floor
  // OGCG: call <4 x float> @llvm.experimental.constrained.floor.v4f32(<4 x float> %{{.*}}, metadata !"fpexcept.ignore")
  return _mm_round_ps(x, 0b1001);
}

__m128 test_mm_round_ps_mxcsr(__m128 x) {
  // CIR-LABEL: test_mm_round_ps_mxcsr
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.ps" %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !s32i) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_round_ps_mxcsr
  // LLVM: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 12)

  // OGCG-LABEL: test_mm_round_ps_mxcsr
  // OGCG: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 12)
  return _mm_round_ps(x, 0b1100);
}

__m128 test_mm_round_ps_fround_no_exc(__m128 x) {
  // CIR-LABEL: test_mm_round_ps_fround_no_exc
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.ps" %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !s32i) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_round_ps_fround_no_exc
  // LLVM: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 0)

  // OGCG-LABEL: test_mm_round_ps_fround_no_exc
  // OGCG: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 0)
  return _mm_round_ps(x, 0b0000);
}

__m128d test_mm_round_sd_ceil(__m128d x, __m128d y) {
  // CIR-LABEL: test_mm_round_sd_ceil
  // %[[A:.*]] = cir.vec.extract = %{{.*}}[%{{.*}} : !u64] : !cir.vector<2 x !cir.double>
  // %[[B:.*]] = cir.ceil %[[A]] : !cir.double {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
  // cir.vec.insert = %[[B]], %{{.*}}[%{{.*}} : !u64] : !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_round_sd_ceil
  // LLVM: %[[A:.*]] = extractelement <2 x double> %{{.*}}, i64 0
  // LLVM: %[[B:.*]] = call double @llvm.experimental.constrained.ceil.f64(double %[[A]], metadata !"fpexcept.ignore")
  // LLVM: insertelement <2 x double> %{{.*}}, double %[[B]], i64 0

  // OGCG-LABEL: test_mm_round_sd_ceil
  // OGCG: %[[A:.*]] = extractelement <2 x double> %{{.*}}, i32 0
  // OGCG: %[[B:.*]] = call double @llvm.experimental.constrained.ceil.f64(double %[[A]], metadata !"fpexcept.ignore")
  // OGCG: insertelement <2 x double> %{{.*}}, double %[[B]], i32 0
  return _mm_round_sd(x, y, 0b1010);
}

__m128d test_mm_round_sd_mxcsr(__m128d x, __m128d y) {
  // CIR-LABEL: test_mm_round_sd_mxcsr
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.sd" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !s32i) -> !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_round_sd_mxcsr
  // LLVM: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 12)

  // OGCG-LABEL: test_mm_round_sd_mxcsr
  // OGCG: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 12)
  return _mm_round_sd(x, y, 0b1100);
}

__m128d test_mm_round_sd_fround_no_exc(__m128d x, __m128d y) {
  // CIR-LABEL: test_mm_round_sd_fround_no_exc
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.sd" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<2 x !cir.double>, !cir.vector<2 x !cir.double>, !s32i) -> !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_round_sd_fround_no_exc
  // LLVM: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 0)

  // OGCG-LABEL: test_mm_round_sd_fround_no_exc
  // OGCG: call <2 x double> @llvm.x86.sse41.round.sd(<2 x double> %{{.*}}, <2 x double> %{{.*}}, i32 0)
  return _mm_round_sd(x, y, 0b0000);
}

__m128 test_mm_round_ss_trunc(__m128 x, __m128 y) {
  // CIR-LABEL: test_mm_round_ss_trunc
  // %[[A:.*]] = cir.vec.extract = %{{.*}}[%{{.*}} : !u64] : !cir.vector<2 x !cir.double>
  // %[[B:.*]] = cir.trunc %6 : !cir.float {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}
  // cir.vec.insert = %[[B]], %{{.*}}[%{{.*}} : !u64] : !cir.vector<2 x !cir.double>

  // LLVM-LABEL: test_mm_round_ss_trunc
  // LLVM: %[[A:.*]] = extractelement <4 x float> %{{.*}}, i64 0
  // LLVM: %[[B:.*]] = call float @llvm.experimental.constrained.trunc.f32(float %[[A]], metadata !"fpexcept.ignore")
  // LLVM: insertelement <4 x float> %{{.*}}, float %[[B]], i64 0

  // OGCG-LABEL: test_mm_round_ss_trunc
  // OGCG: %[[A:.*]] = extractelement <4 x float> %{{.*}}, i32 0
  // OGCG: %[[B:.*]] = call float @llvm.experimental.constrained.trunc.f32(float %[[A]], metadata !"fpexcept.ignore")
  // OGCG: insertelement <4 x float> %{{.*}}, float %[[B]], i32 0
  return _mm_round_ss(x, y, 0b1011);
}

__m128 test_mm_round_ss_mxcsr(__m128 x, __m128 y) {
  // CIR-LABEL: test_mm_round_ss_mxcsr
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.ss" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !s32i) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_round_ss_mxcsr
  // LLVM: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 12)

  // OGCG-LABEL: test_mm_round_ss_mxcsr
  // OGCG: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 12)
  return _mm_round_ss(x, y, 0b1100);
}

__m128 test_mm_round_ss_fround_no_exc(__m128 x, __m128 y) {
  // CIR-LABEL: test_mm_round_ss_fround_no_exc
  // CIR: cir.call_llvm_intrinsic "x86.sse41.round.ss" %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.float>, !cir.vector<4 x !cir.float>, !s32i) -> !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_round_ss_fround_no_exc
  // LLVM: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0)

  // OGCG-LABEL: test_mm_round_ss_fround_no_exc
  // OGCG: call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %{{.*}}, <4 x float> %{{.*}}, i32 0)
  return _mm_round_ss(x, y, 0b0000);
}
