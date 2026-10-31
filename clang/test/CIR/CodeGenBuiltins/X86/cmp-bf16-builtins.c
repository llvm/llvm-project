// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -target-feature +avx10.2 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -target-feature +avx10.2 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefix=CHECK --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -target-feature +avx10.2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefix=CHECK

// Strict FP is not tested: classic codegen hits an unreachable for the bf16
// compare builtins under -ffp-exception-behavior=strict.

#include <immintrin.h>

__mmask8 test_mm_cmp_pbh_mask(__m128bh a, __m128bh b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm_cmp_pbh_mask
  // CIR: cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<8 x !cir.bf16>, !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast

  // CHECK-LABEL: @test_mm_cmp_pbh_mask
  // CHECK: %[[CMP:.*]] = fcmp olt <8 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: bitcast <8 x i1> %[[CMP]] to i8
  return _mm_cmp_pbh_mask(a, b, _CMP_LT_OS);
}

__mmask16 test_mm256_cmp_pbh_mask(__m256bh a, __m256bh b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm256_cmp_pbh_mask
  // CIR: cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<16 x !cir.bf16>, !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast

  // CHECK-LABEL: @test_mm256_cmp_pbh_mask
  // CHECK: %[[CMP:.*]] = fcmp olt <16 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: bitcast <16 x i1> %[[CMP]] to i16
  return _mm256_cmp_pbh_mask(a, b, _CMP_LT_OS);
}

__mmask32 test_mm512_cmp_pbh_mask(__m512bh a, __m512bh b) {
  // CIR-LABEL: cir.func {{.*}}@test_mm512_cmp_pbh_mask
  // CIR: cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<32 x !cir.bf16>, !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast

  // CHECK-LABEL: @test_mm512_cmp_pbh_mask
  // CHECK: %[[CMP:.*]] = fcmp olt <32 x bfloat> %{{.*}}, %{{.*}}
  // CHECK: bitcast <32 x i1> %[[CMP]] to i32
  return _mm512_cmp_pbh_mask(a, b, _CMP_LT_OS);
}
