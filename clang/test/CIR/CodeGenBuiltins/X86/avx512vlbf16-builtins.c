// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -target-feature +avx512bf16 -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -target-feature +avx512bf16 -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -target-feature +avx512bf16 -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=OGCG --input-file=%t.ll %s

#include <immintrin.h>

__m256bh test_mm512_mask_cvtneps_pbh(__m256bh src, __mmask16 k, __m512 a) {
  // CIR-LABEL: test_mm512_mask_cvtneps_pbh
  // CIR: cir.call @_mm512_mask_cvtneps_pbh({{.+}}, {{.+}}, {{.+}}) : (!cir.vector<16 x !cir.bf16>, !u16i, !cir.vector<16 x !cir.float>) -> !cir.vector<16 x !cir.bf16>

  // LLVM-LABEL: @test_mm512_mask_cvtneps_pbh
  // LLVM: call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512

  // OGCG-LABEL: @test_mm512_mask_cvtneps_pbh
  // OGCG: call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512
  return _mm512_mask_cvtneps_pbh(src, k, a);
}

__m256bh test_mm512_maskz_cvtneps_pbh(__mmask16 k, __m512 a) {
  // CIR-LABEL: test_mm512_maskz_cvtneps_pbh
  // CIR: cir.call @_mm512_maskz_cvtneps_pbh({{.+}}, {{.+}}) : (!u16i, !cir.vector<16 x !cir.float>) -> !cir.vector<16 x !cir.bf16>

  // LLVM-LABEL: @test_mm512_maskz_cvtneps_pbh
  // LLVM: call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512(<16 x float> {{.+}})

  // OGCG-LABEL:  @test_mm512_maskz_cvtneps_pbh
  // OGCG: call <16 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.512(<16 x float> {{.+}})
  return _mm512_maskz_cvtneps_pbh(k, a);
}


__m128bh test_mm256_mask_cvtneps_pbh(__m128bh src, __mmask8 k, __m256 a) {
  // CIR-LABEL: test_mm256_mask_cvtneps_pbh
  // CIR: cir.call @_mm256_mask_cvtneps_pbh({{.+}}, {{.+}}, {{.+}}) : (!cir.vector<8 x !cir.bf16>, !u8i, !cir.vector<8 x !cir.float>) -> !cir.vector<8 x !cir.bf16>
  
  // LLVM-LABEL: @test_mm256_mask_cvtneps_pbh
  // LLVM: call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256(<8 x float> {{.+}})

  // OGCG-LABEL: @test_mm256_mask_cvtneps_pbh
  // OGCG: call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256(<8 x float> {{.+}})  
  return _mm256_mask_cvtneps_pbh(src, k, a);
}

__m128bh test_mm256_maskz_cvtneps_pbh(__mmask8 k, __m256 a) {
  // CIR-LABEL: test_mm256_maskz_cvtneps_pbh
  // CIR: cir.call @_mm256_maskz_cvtneps_pbh({{.+}}, {{.+}}) : (!u8i, !cir.vector<8 x !cir.float>) -> !cir.vector<8 x !cir.bf16>

  // LLVM-LABEL: @test_mm256_maskz_cvtneps_pbh
  // LLVM: call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256(<8 x float> {{.+}})

  // OGCG-LABEL: @test_mm256_maskz_cvtneps_pbh
  // OGCG: call <8 x bfloat> @llvm.x86.avx512bf16.cvtneps2bf16.256(<8 x float> {{.+}})
  return _mm256_maskz_cvtneps_pbh(k, a);
}

__m128bh test_mm_mask_cvtneps_pbh(__m128bh src, __mmask8 k, __m128 a) {
  // CIR-LABEL: test_mm_mask_cvtneps_pbh
  // CIR: cir.call @_mm_mask_cvtneps_pbh({{.+}}, {{.+}}, {{.+}}) : (!cir.vector<8 x !cir.bf16>, !u8i, !cir.vector<4 x !cir.float>) -> !cir.vector<8 x !cir.bf16>{{.+}}

  // LLVM-LABEL: @test_mm_mask_cvtneps_pbh
  // LLVM: call <8 x bfloat> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float> {{.+}}, <8 x bfloat> {{.+}}, <4 x i1> {{.+}})

  // OGCG-LABEL: @test_mm_mask_cvtneps_pbh
  // OGCG: call <8 x bfloat> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float> {{.+}}, <8 x bfloat> {{.+}}, <4 x i1> {{.+}})
  return _mm_mask_cvtneps_pbh(src, k, a);
}

__m128bh test_mm_maskz_cvtneps_pbh(__mmask8 k, __m128 a) {
  // CIR-LABEL: test_mm_maskz_cvtneps_pbh
  // CIR: cir.call @_mm_maskz_cvtneps_pbh({{.+}}, {{.+}}) : (!u8i, !cir.vector<4 x !cir.float>) -> !cir.vector<8 x !cir.bf16>
  
  // LLVM-LABEL: @test_mm_maskz_cvtneps_pbh
  // LLVM: call <8 x bfloat> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float> {{.+}}, <8 x bfloat> {{.+}}, <4 x i1> {{.+}})

  // OGCG-LABEL: @test_mm_maskz_cvtneps_pbh
  // OGCG: call <8 x bfloat> @llvm.x86.avx512bf16.mask.cvtneps2bf16.128(<4 x float> {{.+}}, <8 x bfloat> {{.+}}, <4 x i1> {{.+}})
  return _mm_maskz_cvtneps_pbh(k, a);
}
