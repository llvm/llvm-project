// RUN: %clang_cc1 -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefixes=OGCG --input-file=%t.ll %s

#include <immintrin.h>

__m256d test_mm256_round_pd_mxcsr(__m256d x) {
  // CIR-LABEL: test_mm256_round_pd_mxcsr
  // CIR: cir.call_llvm_intrinsic "x86.avx.round.pd.256" %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.double>, !s32i) -> !cir.vector<4 x !cir.double>

  // LLVM-LABEL: test_mm256_round_pd_mxcsr
  // LLVM: call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %{{.*}}, i32 12)

  // OGCG-LABEL: test_mm256_round_pd_mxcsr
  // OGCG: call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %{{.*}}, i32 12)
  return _mm256_round_pd(x, 0b1100);
}

__m256d test_mm256_round_pd_fround_no_exc(__m256d x) {
  // CIR-LABEL: test_mm256_round_pd_fround_no_exc
  // CIR: cir.call_llvm_intrinsic "x86.avx.round.pd.256" %{{.*}}, %{{.*}} : (!cir.vector<4 x !cir.double>, !s32i) -> !cir.vector<4 x !cir.double>

  // LLVM-LABEL: test_mm256_round_pd_fround_no_exc
  // LLVM: call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %{{.*}}, i32 0)

  // OGCG-LABEL: test_mm256_round_pd_fround_no_exc
  // OGCG: call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %{{.*}}, i32 0)
  return _mm256_round_pd(x, 0b0000);
}

__m256d test_mm256_round_pd_trunc(__m256d x) {
  // CIR-LABEL: test_mm256_round_pd_trunc
  // CIR: cir.trunc %{{.*}} : !cir.vector<4 x !cir.double> {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}

  // LLVM-LABEL: test_mm256_round_pd_trunc
  // LLVM: call <4 x double> @llvm.experimental.constrained.trunc.v4f64(<4 x double> %{{.*}}, metadata !"fpexcept.ignore")

  // OGCG-LABEL: test_mm256_round_pd_trunc
  // OGCG: call <4 x double> @llvm.experimental.constrained.trunc.v4f64(<4 x double> %{{.*}}, metadata !"fpexcept.ignore")
  return _mm256_round_pd(x, 0b1011);
}

__m256 test_mm256_round_ps_mxcsr(__m256 x) {
  // CIR-LABEL: test_mm256_round_ps_mxcsr
  // CIR: cir.call_llvm_intrinsic "x86.avx.round.ps.256" %{{.*}}, %{{.*}} : (!cir.vector<8 x !cir.float>, !s32i) -> !cir.vector<8 x !cir.float>

  // LLVM-LABEL: test_mm256_round_ps_mxcsr
  // LLVM: call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %{{.*}}, i32 12)

  // OGCG-LABEL: test_mm256_round_ps_mxcsr
  // OGCG: call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %{{.*}}, i32 12)
  return _mm256_round_ps(x, 0b1100);
}

__m256 test_mm256_round_ps_fround_no_exc(__m256 x) {
  // CIR-LABEL: test_mm256_round_ps_fround_no_exc
  // CIR: cir.call_llvm_intrinsic "x86.avx.round.ps.256" %{{.*}}, %{{.*}} : (!cir.vector<8 x !cir.float>, !s32i) -> !cir.vector<8 x !cir.float>

  // LLVM-LABEL: test_mm256_round_ps_fround_no_exc
  // LLVM: call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %{{.*}}, i32 0)

  // OGCG-LABEL: test_mm256_round_ps_fround_no_exc
  // OGCG: call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %{{.*}}, i32 0)
  return _mm256_round_ps(x, 0b0000);
}

__m256 test_mm256_round_ps_trunc(__m256 x) {
  // CIR-LABEL: test_mm256_round_ps_trunc
  // CIR: cir.trunc %{{.*}} : !cir.vector<8 x !cir.float> {fenv = #cir.fenv<dynamic_rounding_mode = unknown, except_mode = masked, strict_except = false>}

  // LLVM-LABEL: test_mm256_round_ps_trunc
  // LLVM: call <8 x float> @llvm.experimental.constrained.trunc.v8f32(<8 x float> %{{.*}}, metadata !"fpexcept.ignore")

  // OGCG-LABEL: test_mm256_round_ps_trunc
  // OGCG: call <8 x float> @llvm.experimental.constrained.trunc.v8f32(<8 x float> %{{.*}}, metadata !"fpexcept.ignore")
  return _mm256_round_ps(x, 0b1011);
}
