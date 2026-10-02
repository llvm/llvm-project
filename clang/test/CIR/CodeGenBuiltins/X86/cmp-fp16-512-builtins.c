// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -target-feature +avx512fp16 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -target-feature +avx512fp16 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefix=CHECK --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -target-feature +avx512fp16 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -target-feature +avx512fp16 -ffp-exception-behavior=strict -fexperimental-strict-floating-point -fclangir -emit-cir -o %t.strict.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR-STRICT --input-file=%t.strict.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -target-feature +avx512fp16 -ffp-exception-behavior=strict -fexperimental-strict-floating-point -fclangir -emit-llvm -o %t.strict.ll -Wall -Werror
// RUN: FileCheck --check-prefix=CHECK-STRICT --input-file=%t.strict.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -target-feature +avx512fp16 -ffp-exception-behavior=strict -fexperimental-strict-floating-point -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK-STRICT

#include <immintrin.h>

__mmask8 test_mm_cmp_ph_mask(__m128h a, __m128h b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm_cmp_ph_mask
  // CIR: cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<8 x !cir.f16>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast

  // CHECK-LABEL: @test_mm_cmp_ph_mask
  // CHECK: %[[CMP:.*]] = fcmp olt <8 x half> %{{.*}}, %{{.*}}
  // CHECK: bitcast <8 x i1> %[[CMP]] to i8

  // CIR-STRICT-LABEL: cir.func {{.*}}@test_mm_cmp_ph_mask
  // CIR-STRICT: cir.call_llvm_intrinsic "x86.avx512fp16.mask.cmp.ph.128"

  // CHECK-STRICT-LABEL: @test_mm_cmp_ph_mask
  // CHECK-STRICT: call <8 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.128(<8 x half> %{{.*}}, <8 x half> %{{.*}}, i32 1, <8 x i1> splat (i1 true))
  return _mm_cmp_ph_mask(a, b, _CMP_LT_OS);
}

__mmask16 test_mm256_cmp_ph_mask(__m256h a, __m256h b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm256_cmp_ph_mask
  // CIR: cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<16 x !cir.f16>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast

  // CHECK-LABEL: @test_mm256_cmp_ph_mask
  // CHECK: %[[CMP:.*]] = fcmp olt <16 x half> %{{.*}}, %{{.*}}
  // CHECK: bitcast <16 x i1> %[[CMP]] to i16

  // CIR-STRICT-LABEL: cir.func {{.*}}@test_mm256_cmp_ph_mask
  // CIR-STRICT: cir.call_llvm_intrinsic "x86.avx512fp16.mask.cmp.ph.256"

  // CHECK-STRICT-LABEL: @test_mm256_cmp_ph_mask
  // CHECK-STRICT: call <16 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.256(<16 x half> %{{.*}}, <16 x half> %{{.*}}, i32 1, <16 x i1> splat (i1 true))
  return _mm256_cmp_ph_mask(a, b, _CMP_LT_OS);
}

__mmask32 test_mm512_cmp_ph_mask(__m512h a, __m512h b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm512_cmp_ph_mask
  // CIR: cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<32 x !cir.f16>, !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast

  // CHECK-LABEL: @test_mm512_cmp_ph_mask
  // CHECK: %[[CMP:.*]] = fcmp olt <32 x half> %{{.*}}, %{{.*}}
  // CHECK: bitcast <32 x i1> %[[CMP]] to i32

  // CIR-STRICT-LABEL: cir.func {{.*}}@test_mm512_cmp_ph_mask
  // CIR-STRICT: cir.call_llvm_intrinsic "x86.avx512fp16.mask.cmp.ph.512"

  // CHECK-STRICT-LABEL: @test_mm512_cmp_ph_mask
  // CHECK-STRICT: call <32 x i1> @llvm.x86.avx512fp16.mask.cmp.ph.512(<32 x half> %{{.*}}, <32 x half> %{{.*}}, i32 1, <32 x i1> splat (i1 true), i32 4)
  return _mm512_cmp_ph_mask(a, b, _CMP_LT_OS);
}

__mmask16 test_mm512_cmp_ps_mask(__m512 a, __m512 b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm512_cmp_ps_mask
  // CIR: cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<16 x !cir.float>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast

  // CHECK-LABEL: @test_mm512_cmp_ps_mask
  // CHECK: %[[CMP:.*]] = fcmp olt <16 x float> %{{.*}}, %{{.*}}
  // CHECK: bitcast <16 x i1> %[[CMP]] to i16

  // CIR-STRICT-LABEL: cir.func {{.*}}@test_mm512_cmp_ps_mask
  // CIR-STRICT: cir.call_llvm_intrinsic "x86.avx512.mask.cmp.ps.512"

  // CHECK-STRICT-LABEL: @test_mm512_cmp_ps_mask
  // CHECK-STRICT: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 1, <16 x i1> splat (i1 true), i32 4)
  return _mm512_cmp_ps_mask(a, b, _CMP_LT_OS);
}

__mmask8 test_mm512_cmp_pd_mask(__m512d a, __m512d b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm512_cmp_pd_mask
  // CIR: cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<8 x !cir.double>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast

  // CHECK-LABEL: @test_mm512_cmp_pd_mask
  // CHECK: %[[CMP:.*]] = fcmp olt <8 x double> %{{.*}}, %{{.*}}
  // CHECK: bitcast <8 x i1> %[[CMP]] to i8

  // CIR-STRICT-LABEL: cir.func {{.*}}@test_mm512_cmp_pd_mask
  // CIR-STRICT: cir.call_llvm_intrinsic "x86.avx512.mask.cmp.pd.512"

  // CHECK-STRICT-LABEL: @test_mm512_cmp_pd_mask
  // CHECK-STRICT: call <8 x i1> @llvm.x86.avx512.mask.cmp.pd.512(<8 x double> %{{.*}}, <8 x double> %{{.*}}, i32 1, <8 x i1> splat (i1 true), i32 4)
  return _mm512_cmp_pd_mask(a, b, _CMP_LT_OS);
}

// The explicit rounding/SAE argument is passed through to the intrinsic only
// under strict FP.
__mmask16 test_mm512_cmp_round_ps_mask(__m512 a, __m512 b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm512_cmp_round_ps_mask
  // CIR: cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<16 x !cir.float>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast

  // CHECK-LABEL: @test_mm512_cmp_round_ps_mask
  // CHECK: %[[CMP:.*]] = fcmp olt <16 x float> %{{.*}}, %{{.*}}
  // CHECK: bitcast <16 x i1> %[[CMP]] to i16

  // CIR-STRICT-LABEL: cir.func {{.*}}@test_mm512_cmp_round_ps_mask
  // CIR-STRICT: cir.call_llvm_intrinsic "x86.avx512.mask.cmp.ps.512"

  // CHECK-STRICT-LABEL: @test_mm512_cmp_round_ps_mask
  // CHECK-STRICT: call <16 x i1> @llvm.x86.avx512.mask.cmp.ps.512(<16 x float> %{{.*}}, <16 x float> %{{.*}}, i32 1, <16 x i1> splat (i1 true), i32 8)
  return _mm512_cmp_round_ps_mask(a, b, _CMP_LT_OS, _MM_FROUND_NO_EXC);
}
