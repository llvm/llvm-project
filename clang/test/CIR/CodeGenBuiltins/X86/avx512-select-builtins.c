// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2 -fclangir -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2 -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2 -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefixes=OGCG --input-file=%t.ll %s

#include <immintrin.h>


__m128i test_selectb_128(__mmask16 k, __m128i a, __m128i b) {
  // CIR-LABEL: @test_selectb_128
  // CIR: %[[MASK_BC:.+]] = cir.cast bitcast %{{.+}} : !u16i -> !cir.vector<16 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary(%[[MASK_BC]], %{{.+}}, %{{.+}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !s8i>

  // LLVM-LABEL: @test_selectb_128
  // LLVM: select <16 x i1> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8> %{{.+}}

  // OGCG-LABEL: @test_selectb_128
  // OGCG: select <16 x i1> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8> %{{.+}}
  return (__m128i)__builtin_ia32_selectb_128(k, (__v16qi)a, (__v16qi)b);
}

__m256i test_selectb_256(__mmask32 k, __m256i a, __m256i b) {
  // CIR-LABEL: @test_selectb_256
  // CIR: %[[MASK_BC:.+]] = cir.cast bitcast %{{.+}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary(%[[MASK_BC]], %{{.+}}, %{{.+}}) : !cir.vector<32 x !cir.int<s, 1>>, !cir.vector<32 x !s8i>

  // LLVM-LABEL: @test_selectb_256
  // LLVM: select <32 x i1> %{{.+}}, <32 x i8> %{{.+}}, <32 x i8> %{{.+}}

  // OGCG-LABEL: @test_selectb_256
  // OGCG: select <32 x i1> %{{.+}}, <32 x i8> %{{.+}}, <32 x i8> %{{.+}}
  return (__m256i)__builtin_ia32_selectb_256(k, (__v32qi)a, (__v32qi)b);
}

__m512i test_selectb_512(__mmask64 k, __m512i a, __m512i b) {
  // CIR-LABEL: @test_selectb_512
  // CIR: %[[MASK_BC:.+]] = cir.cast bitcast %{{.+}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary(%[[MASK_BC]], %{{.+}}, %{{.+}}) : !cir.vector<64 x !cir.int<s, 1>>, !cir.vector<64 x !s8i>

  // LLVM-LABEL: @test_selectb_512
  // LLVM: select <64 x i1> %{{.+}}, <64 x i8> %{{.+}}, <64 x i8> %{{.+}}

  // OGCG-LABEL: @test_selectb_512
  // OGCG: select <64 x i1> %{{.+}}, <64 x i8> %{{.+}}, <64 x i8> %{{.+}}
  return (__m512i)__builtin_ia32_selectb_512(k, (__v64qi)a, (__v64qi)b);
}

__m128i test_selectw_128(__mmask8 k, __m128i a, __m128i b) {
  // CIR-LABEL: @test_selectw_128
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !s16i>

  // LLVM-LABEL: @test_selectw_128
  // LLVM: select <8 x i1> %{{.+}}, <8 x i16> %{{.+}}, <8 x i16> %{{.+}}

  // OGCG-LABEL: @test_selectw_128
  // OGCG: select <8 x i1> %{{.+}}, <8 x i16> %{{.+}}, <8 x i16> %{{.+}}
  return (__m128i)__builtin_ia32_selectw_128(k, (__v8hi)a, (__v8hi)b);
}

__m256i test_selectw_256(__mmask16 k, __m256i a, __m256i b) {
  // CIR-LABEL: @test_selectw_256
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !s16i>

  // LLVM-LABEL: @test_selectw_256
  // LLVM: select <16 x i1> %{{.+}}, <16 x i16> %{{.+}}, <16 x i16> %{{.+}}

  // OGCG-LABEL: @test_selectw_256
  // OGCG: select <16 x i1> %{{.+}}, <16 x i16> %{{.+}}, <16 x i16> %{{.+}}
  return (__m256i)__builtin_ia32_selectw_256(k, (__v16hi)a, (__v16hi)b);
}

__m512i test_selectw_512(__mmask32 k, __m512i a, __m512i b) {
  // CIR-LABEL: @test_selectw_512
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<32 x !cir.int<s, 1>>, !cir.vector<32 x !s16i>

  // LLVM-LABEL: @test_selectw_512
  // LLVM: select <32 x i1> %{{.+}}, <32 x i16> %{{.+}}, <32 x i16> %{{.+}}

  // OGCG-LABEL: @test_selectw_512
  // OGCG: select <32 x i1> %{{.+}}, <32 x i16> %{{.+}}, <32 x i16> %{{.+}}
  return (__m512i)__builtin_ia32_selectw_512(k, (__v32hi)a, (__v32hi)b);
}

__m128i test_selectd_128(__mmask8 k, __m128i a, __m128i b) {
  // CIR-LABEL: @test_selectd_128
  // CIR: %[[M_SHUF:.+]] = cir.vec.shuffle(%{{.+}}, %{{.+}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]
  // CIR: cir.vec.ternary(%[[M_SHUF]], %{{.+}}, %{{.+}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !s32i>

  // LLVM-LABEL: @test_selectd_128
  // LLVM: shufflevector <8 x i1> %{{.+}}, <8 x i1> %{{.+}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM: select <4 x i1> %{{.+}}, <4 x i32> %{{.+}}, <4 x i32> %{{.+}}

  // OGCG-LABEL: @test_selectd_128
  // OGCG: select <4 x i1> %{{.+}}, <4 x i32> %{{.+}}, <4 x i32> %{{.+}}
  return (__m128i)__builtin_ia32_selectd_128(k, (__v4si)a, (__v4si)b);
}

__m256i test_selectd_256(__mmask8 k, __m256i a, __m256i b) {
  // CIR-LABEL: @test_selectd_256
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !s32i>

  // LLVM-LABEL: @test_selectd_256
  // LLVM: select <8 x i1> %{{.+}}, <8 x i32> %{{.+}}, <8 x i32> %{{.+}}

  // OGCG-LABEL: @test_selectd_256
  // OGCG: select <8 x i1> %{{.+}}, <8 x i32> %{{.+}}, <8 x i32> %{{.+}}
  return (__m256i)__builtin_ia32_selectd_256(k, (__v8si)a, (__v8si)b);
}

__m512i test_selectd_512(__mmask16 k, __m512i a, __m512i b) {
  // CIR-LABEL: @test_selectd_512
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !s32i>

  // LLVM-LABEL: @test_selectd_512
  // LLVM: select <16 x i1> %{{.+}}, <16 x i32> %{{.+}}, <16 x i32> %{{.+}}

  // OGCG-LABEL: @test_selectd_512
  // OGCG: select <16 x i1> %{{.+}}, <16 x i32> %{{.+}}, <16 x i32> %{{.+}}
  return (__m512i)__builtin_ia32_selectd_512(k, (__v16si)a, (__v16si)b);
}

__m128i test_selectq_128(__mmask8 k, __m128i a, __m128i b) {
  // CIR-LABEL: @test_selectq_128
  // CIR: %[[M_SHUF:.+]] = cir.vec.shuffle(%{{.+}}, %{{.+}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i]
  // CIR: cir.vec.ternary(%[[M_SHUF]], %{{.+}}, %{{.+}}) : !cir.vector<2 x !cir.int<s, 1>>, !cir.vector<2 x !s64i>

  // LLVM-LABEL: @test_selectq_128
  // LLVM: select <2 x i1> %{{.+}}, <2 x i64> %{{.+}}, <2 x i64> %{{.+}}

  // OGCG-LABEL: @test_selectq_128
  // OGCG: select <2 x i1> %{{.+}}, <2 x i64> %{{.+}}, <2 x i64> %{{.+}}
  return __builtin_ia32_selectq_128(k, (__v2di)a, (__v2di)b);
}

__m256i test_selectq_256(__mmask8 k, __m256i a, __m256i b) {
  // CIR-LABEL: @test_selectq_256
  // CIR: %[[M_SHUF:.+]] = cir.vec.shuffle(%{{.+}}, %{{.+}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]
  // CIR: cir.vec.ternary(%[[M_SHUF]], %{{.+}}, %{{.+}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !s64i>

  // LLVM-LABEL: @test_selectq_256
  // LLVM: select <4 x i1> %{{.+}}, <4 x i64> %{{.+}}, <4 x i64> %{{.+}}

  // OGCG-LABEL: @test_selectq_256
  // OGCG: select <4 x i1> %{{.+}}, <4 x i64> %{{.+}}, <4 x i64> %{{.+}}
  return __builtin_ia32_selectq_256(k, (__v4di)a, (__v4di)b);
}

__m512i test_selectq_512(__mmask8 k, __m512i a, __m512i b) {
  // CIR-LABEL: @test_selectq_512
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !s64i>

  // LLVM-LABEL: @test_selectq_512
  // LLVM: select <8 x i1> %{{.+}}, <8 x i64> %{{.+}}, <8 x i64> %{{.+}}

  // OGCG-LABEL: @test_selectq_512
  // OGCG: select <8 x i1> %{{.+}}, <8 x i64> %{{.+}}, <8 x i64> %{{.+}}
  return __builtin_ia32_selectq_512(k, (__v8di)a, (__v8di)b);
}

__m128h test_selectph_128(__mmask8 k, __m128h a, __m128h b) {
  // CIR-LABEL: @test_selectph_128
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !cir.f16>

  // LLVM-LABEL: @test_selectph_128
  // LLVM: select <8 x i1> %{{.+}}, <8 x half> %{{.+}}, <8 x half> %{{.+}}

  // OGCG-LABEL: @test_selectph_128
  // OGCG: select <8 x i1> %{{.+}}, <8 x half> %{{.+}}, <8 x half> %{{.+}}
  return __builtin_ia32_selectph_128(k, (__v8hf)a, (__v8hf)b);
}

__m256h test_selectph_256(__mmask16 k, __m256h a, __m256h b) {
  // CIR-LABEL: @test_selectph_256
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !cir.f16>

  // LLVM-LABEL: @test_selectph_256
  // LLVM: select <16 x i1> %{{.+}}, <16 x half> %{{.+}}, <16 x half> %{{.+}}

  // OGCG-LABEL: @test_selectph_256
  // OGCG: select <16 x i1> %{{.+}}, <16 x half> %{{.+}}, <16 x half> %{{.+}}
  return __builtin_ia32_selectph_256(k, (__v16hf)a, (__v16hf)b);
}

__m512h test_selectph_512(__mmask32 k, __m512h a, __m512h b) {
  // CIR-LABEL: @test_selectph_512
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<32 x !cir.int<s, 1>>, !cir.vector<32 x !cir.f16>

  // LLVM-LABEL: @test_selectph_512
  // LLVM: select <32 x i1> %{{.+}}, <32 x half> %{{.+}}, <32 x half> %{{.+}}

  // OGCG-LABEL: @test_selectph_512
  // OGCG: select <32 x i1> %{{.+}}, <32 x half> %{{.+}}, <32 x half> %{{.+}}
  return __builtin_ia32_selectph_512(k, (__v32hf)a, (__v32hf)b);
}

__m256bh test_selectpbf_256(__mmask16 k, __m256bh a, __m256bh b) {
  // CIR-LABEL: @test_selectpbf_256
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !cir.bf16>

  // LLVM-LABEL: @test_selectpbf_256
  // LLVM: select <16 x i1> %{{.+}}, <16 x bfloat> %{{.+}}, <16 x bfloat> %{{.+}}

  // OGCG-LABEL: @test_selectpbf_256
  // OGCG: select <16 x i1> %{{.+}}, <16 x bfloat> %{{.+}}, <16 x bfloat> %{{.+}}
  return __builtin_ia32_selectpbf_256(k, (__v16bf)a, (__v16bf)b);
}

__m512bh test_selectpbf_512(__mmask32 k, __m512bh a, __m512bh b) {
  // CIR-LABEL: @test_selectpbf_512
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<32 x !cir.int<s, 1>>, !cir.vector<32 x !cir.bf16>

  // LLVM-LABEL: @test_selectpbf_512
  // LLVM: select <32 x i1> %{{.+}}, <32 x bfloat> %{{.+}}, <32 x bfloat> %{{.+}}

  // OGCG-LABEL: @test_selectpbf_512
  // OGCG: select <32 x i1> %{{.+}}, <32 x bfloat> %{{.+}}, <32 x bfloat> %{{.+}}
  return __builtin_ia32_selectpbf_512(k, (__v32bf)a, (__v32bf)b);
}

__m128 test_selectps_128(__mmask8 k, __m128 a, __m128 b) {
  // CIR-LABEL: @test_selectps_128
  // CIR: %[[M_SHUF:.+]] = cir.vec.shuffle(%{{.+}}, %{{.+}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]
  // CIR: cir.vec.ternary(%[[M_SHUF]], %{{.+}}, %{{.+}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !cir.float>

  // LLVM-LABEL: @test_selectps_128
  // LLVM: select <4 x i1> %{{.+}}, <4 x float> %{{.+}}, <4 x float> %{{.+}}

  // OGCG-LABEL: @test_selectps_128
  // OGCG: select <4 x i1> %{{.+}}, <4 x float> %{{.+}}, <4 x float> %{{.+}}
  return __builtin_ia32_selectps_128(k, (__v4sf)a, (__v4sf)b);
}

__m256 test_selectps_256(__mmask8 k, __m256 a, __m256 b) {
  // CIR-LABEL: @test_selectps_256
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !cir.float>

  // LLVM-LABEL: @test_selectps_256
  // LLVM: select <8 x i1> %{{.+}}, <8 x float> %{{.+}}, <8 x float> %{{.+}}

  // OGCG-LABEL: @test_selectps_256
  // OGCG: select <8 x i1> %{{.+}}, <8 x float> %{{.+}}, <8 x float> %{{.+}}
  return __builtin_ia32_selectps_256(k, (__v8sf)a, (__v8sf)b);
}

__m512 test_selectps_512(__mmask16 k, __m512 a, __m512 b) {
  // CIR-LABEL: @test_selectps_512
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<16 x !cir.int<s, 1>>, !cir.vector<16 x !cir.float>

  // LLVM-LABEL: @test_selectps_512
  // LLVM: select <16 x i1> %{{.+}}, <16 x float> %{{.+}}, <16 x float> %{{.+}}

  // OGCG-LABEL: @test_selectps_512
  // OGCG: select <16 x i1> %{{.+}}, <16 x float> %{{.+}}, <16 x float> %{{.+}}
  return __builtin_ia32_selectps_512(k, (__v16sf)a, (__v16sf)b);
}

__m128d test_selectpd_128(__mmask8 k, __m128d a, __m128d b) {
  // CIR-LABEL: @test_selectpd_128
  // CIR: %[[M_SHUF:.+]] = cir.vec.shuffle(%{{.+}}, %{{.+}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i]
  // CIR: cir.vec.ternary(%[[M_SHUF]], %{{.+}}, %{{.+}}) : !cir.vector<2 x !cir.int<s, 1>>, !cir.vector<2 x !cir.double>

  // LLVM-LABEL: @test_selectpd_128
  // LLVM: select <2 x i1> %{{.+}}, <2 x double> %{{.+}}, <2 x double> %{{.+}}

  // OGCG-LABEL: @test_selectpd_128
  // OGCG: select <2 x i1> %{{.+}}, <2 x double> %{{.+}}, <2 x double> %{{.+}}
  return __builtin_ia32_selectpd_128(k, (__v2df)a, (__v2df)b);
}

__m256d test_selectpd_256(__mmask8 k, __m256d a, __m256d b) {
  // CIR-LABEL: @test_selectpd_256
  // CIR: %[[M_SHUF:.+]] = cir.vec.shuffle(%{{.+}}, %{{.+}} : !cir.vector<8 x !cir.int<s, 1>>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]
  // CIR: cir.vec.ternary(%[[M_SHUF]], %{{.+}}, %{{.+}}) : !cir.vector<4 x !cir.int<s, 1>>, !cir.vector<4 x !cir.double>

  // LLVM-LABEL: @test_selectpd_256
  // LLVM: select <4 x i1> %{{.+}}, <4 x double> %{{.+}}, <4 x double> %{{.+}}

  // OGCG-LABEL: @test_selectpd_256
  // OGCG: select <4 x i1> %{{.+}}, <4 x double> %{{.+}}, <4 x double> %{{.+}}
  return __builtin_ia32_selectpd_256(k, (__v4df)a, (__v4df)b);
}

__m512d test_selectpd_512(__mmask8 k, __m512d a, __m512d b) {
  // CIR-LABEL: @test_selectpd_512
  // CIR: cir.vec.ternary(%{{.+}}, %{{.+}}, %{{.+}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !cir.double>

  // LLVM-LABEL: @test_selectpd_512
  // LLVM: select <8 x i1> %{{.+}}, <8 x double> %{{.+}}, <8 x double> %{{.+}}

  // OGCG-LABEL: @test_selectpd_512
  // OGCG: select <8 x i1> %{{.+}}, <8 x double> %{{.+}}, <8 x double> %{{.+}}
  return __builtin_ia32_selectpd_512(k, (__v8df)a, (__v8df)b);
}

// Scalar Selects 

__m128h test_selectsh_128(__mmask8 k, __m128h a, __m128h b) {
  // CIR-LABEL: @test_selectsh_128
  // CIR: %[[A:.+]] = cir.load align(16) %{{.+}} : !cir.ptr<!cir.vector<8 x !cir.f16>>, !cir.vector<8 x !cir.f16>
  // CIR: %[[B:.+]] = cir.load align(16) %{{.+}} : !cir.ptr<!cir.vector<8 x !cir.f16>>, !cir.vector<8 x !cir.f16>
  // CIR: %[[IDX0_A:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[EA:.+]] = cir.vec.extract %[[A]][%[[IDX0_A]] : !u64i] : !cir.vector<8 x !cir.f16>
  // CIR: %[[IDX0_B:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[EB:.+]] = cir.vec.extract %[[B]][%[[IDX0_B]] : !u64i] : !cir.vector<8 x !cir.f16>
  // CIR: %[[MASK_VEC:.+]] = cir.cast bitcast %{{.+}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[IDX0_MASK:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[BIT0:.+]] = cir.vec.extract %[[MASK_VEC]][%[[IDX0_MASK]] : !u64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[COND:.+]] = cir.cast int_to_bool %[[BIT0]] : !cir.int<s, 1> -> !cir.bool
  // CIR: %[[SEL:.+]] = cir.select if %[[COND]] then %[[EA]] else %[[EB]] : (!cir.bool, !cir.f16, !cir.f16) -> !cir.f16
  // CIR: %[[IDX0_INS:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %{{.+}} = cir.vec.insert %[[SEL]], %[[A]][%[[IDX0_INS]] : !u64i] : !cir.vector<8 x !cir.f16>

  // LLVM-LABEL: @test_selectsh_128
  // LLVM: %[[E1:.+]] = extractelement <8 x half> %{{.+}}, i64 0
  // LLVM: %[[E2:.+]] = extractelement <8 x half> %{{.+}}, i64 0
  // LLVM: %[[MASK:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // LLVM: %[[BIT:.+]] = extractelement <8 x i1> %[[MASK]], i64 0
  // LLVM: %[[CMP:.+]] = icmp ne i1 %[[BIT]], false
  // LLVM: %[[SEL:.+]] = select i1 %[[CMP]], half %[[E1]], half %[[E2]]
  // LLVM: insertelement <8 x half> %{{.+}}, half %[[SEL]], i64 0

  // OGCG-LABEL: @test_selectsh_128
  // OGCG: %[[E1:.+]] = extractelement <8 x half> %{{.+}}, i64 0
  // OGCG: %[[E2:.+]] = extractelement <8 x half> %{{.+}}, i64 0
  // OGCG: %[[MASK:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // OGCG: %[[BIT:.+]] = extractelement <8 x i1> %[[MASK]], i64 0
  // OGCG: %{{.+}} = select i1 %[[BIT]], half %[[E1]], half %[[E2]]
  // OGCG: insertelement <8 x half> %{{.+}}, half %{{.+}}, i64 0
  return __builtin_ia32_selectsh_128(k, (__v8hf)a, (__v8hf)b);
}

__m128bh test_selectsbf_128(__mmask8 k, __m128bh a, __m128bh b) {
  // CIR-LABEL: @test_selectsbf_128
  // CIR: %[[A:.+]] = cir.load align(16) %{{.+}} : !cir.ptr<!cir.vector<8 x !cir.bf16>>, !cir.vector<8 x !cir.bf16>
  // CIR: %[[B:.+]] = cir.load align(16) %{{.+}} : !cir.ptr<!cir.vector<8 x !cir.bf16>>, !cir.vector<8 x !cir.bf16>
  // CIR: %[[IDX0_A:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[EA:.+]] = cir.vec.extract %[[A]][%[[IDX0_A]] : !u64i] : !cir.vector<8 x !cir.bf16>
  // CIR: %[[IDX0_B:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[EB:.+]] = cir.vec.extract %[[B]][%[[IDX0_B]] : !u64i] : !cir.vector<8 x !cir.bf16>
  // CIR: %[[MASK_VEC:.+]] = cir.cast bitcast %{{.+}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[IDX0_MASK:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[BIT0:.+]] = cir.vec.extract %[[MASK_VEC]][%[[IDX0_MASK]] : !u64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[COND:.+]] = cir.cast int_to_bool %[[BIT0]] : !cir.int<s, 1> -> !cir.bool
  // CIR: %[[SEL:.+]] = cir.select if %[[COND]] then %[[EA]] else %[[EB]] : (!cir.bool, !cir.bf16, !cir.bf16) -> !cir.bf16
  // CIR: %[[IDX0_INS:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %{{.+}} = cir.vec.insert %[[SEL]], %[[A]][%[[IDX0_INS]] : !u64i] : !cir.vector<8 x !cir.bf16>

  // LLVM-LABEL: @test_selectsbf_128
  // LLVM: %[[E1:.+]] = extractelement <8 x bfloat> %{{.+}}, i64 0
  // LLVM: %[[E2:.+]] = extractelement <8 x bfloat> %{{.+}}, i64 0
  // LLVM: %[[MASK:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // LLVM: %[[BIT:.+]] = extractelement <8 x i1> %[[MASK]], i64 0
  // LLVM: %[[CMP:.+]] = icmp ne i1 %[[BIT]], false
  // LLVM: %[[SEL:.+]] = select i1 %[[CMP]], bfloat %[[E1]], bfloat %[[E2]]
  // LLVM: insertelement <8 x bfloat> %{{.+}}, bfloat %[[SEL]], i64 0

  // OGCG-LABEL: @test_selectsbf_128
  // OGCG: %[[E1:.+]] = extractelement <8 x bfloat> %{{.+}}, i64 0
  // OGCG: %[[E2:.+]] = extractelement <8 x bfloat> %{{.+}}, i64 0
  // OGCG: %[[MASK:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // OGCG: %[[BIT:.+]] = extractelement <8 x i1> %[[MASK]], i64 0
  // OGCG: %{{.+}} = select i1 %[[BIT]], bfloat %[[E1]], bfloat %[[E2]]
  // OGCG: insertelement <8 x bfloat> %{{.+}}, bfloat %{{.+}}, i64 0
  return __builtin_ia32_selectsbf_128(k, (__v8bf)a, (__v8bf)b);
}

__m128 test_selectss_128(__mmask8 k, __m128 a, __m128 b) {
  // CIR-LABEL: @test_selectss_128
  // CIR: %[[A:.+]] = cir.load align(16) %{{.+}} : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR: %[[B:.+]] = cir.load align(16) %{{.+}} : !cir.ptr<!cir.vector<4 x !cir.float>>, !cir.vector<4 x !cir.float>
  // CIR: %[[IDX0_A:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[EA:.+]] = cir.vec.extract %[[A]][%[[IDX0_A]] : !u64i] : !cir.vector<4 x !cir.float>
  // CIR: %[[IDX0_B:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[EB:.+]] = cir.vec.extract %[[B]][%[[IDX0_B]] : !u64i] : !cir.vector<4 x !cir.float>
  // CIR: %[[MASK_VEC:.+]] = cir.cast bitcast %{{.+}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[IDX0_MASK:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[BIT0:.+]] = cir.vec.extract %[[MASK_VEC]][%[[IDX0_MASK]] : !u64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[COND:.+]] = cir.cast int_to_bool %[[BIT0]] : !cir.int<s, 1> -> !cir.bool
  // CIR: %[[SEL:.+]] = cir.select if %[[COND]] then %[[EA]] else %[[EB]] : (!cir.bool, !cir.float, !cir.float) -> !cir.float
  // CIR: %[[IDX0_INS:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %{{.+}} = cir.vec.insert %[[SEL]], %[[A]][%[[IDX0_INS]] : !u64i] : !cir.vector<4 x !cir.float>

  // LLVM-LABEL: @test_selectss_128
  // LLVM: %[[E1:.+]] = extractelement <4 x float> %{{.+}}, i64 0
  // LLVM: %[[E2:.+]] = extractelement <4 x float> %{{.+}}, i64 0
  // LLVM: %[[MASK:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // LLVM: %[[BIT:.+]] = extractelement <8 x i1> %[[MASK]], i64 0
  // LLVM: %[[CMP:.+]] = icmp ne i1 %[[BIT]], false
  // LLVM: %[[SEL:.+]] = select i1 %[[CMP]], float %[[E1]], float %[[E2]]
  // LLVM: insertelement <4 x float> %{{.+}}, float %[[SEL]], i64 0

  // OGCG-LABEL: @test_selectss_128
  // OGCG: %[[E1:.+]] = extractelement <4 x float> %{{.+}}, i64 0
  // OGCG: %[[E2:.+]] = extractelement <4 x float> %{{.+}}, i64 0
  // OGCG: %[[MASK:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // OGCG: %[[BIT:.+]] = extractelement <8 x i1> %[[MASK]], i64 0
  // OGCG: %{{.+}} = select i1 %[[BIT]], float %[[E1]], float %[[E2]]
  // OGCG: insertelement <4 x float> %{{.+}}, float %{{.+}}, i64 0
  return __builtin_ia32_selectss_128(k, (__v4sf)a, (__v4sf)b);
}

__m128d test_selectsd_128(__mmask8 k, __m128d a, __m128d b) {
  // CIR-LABEL: @test_selectsd_128
  // CIR: %[[A:.+]] = cir.load align(16) %{{.+}} : !cir.ptr<!cir.vector<2 x !cir.double>>, !cir.vector<2 x !cir.double>
  // CIR: %[[B:.+]] = cir.load align(16) %{{.+}} : !cir.ptr<!cir.vector<2 x !cir.double>>, !cir.vector<2 x !cir.double>
  // CIR: %[[IDX0_A:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[EA:.+]] = cir.vec.extract %[[A]][%[[IDX0_A]] : !u64i] : !cir.vector<2 x !cir.double>
  // CIR: %[[IDX0_B:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[EB:.+]] = cir.vec.extract %[[B]][%[[IDX0_B]] : !u64i] : !cir.vector<2 x !cir.double>
  // CIR: %[[MASK_VEC:.+]] = cir.cast bitcast %{{.+}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[IDX0_MASK:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[BIT0:.+]] = cir.vec.extract %[[MASK_VEC]][%[[IDX0_MASK]] : !u64i] : !cir.vector<8 x !cir.int<s, 1>>
  // CIR: %[[COND:.+]] = cir.cast int_to_bool %[[BIT0]] : !cir.int<s, 1> -> !cir.bool
  // CIR: %[[SEL:.+]] = cir.select if %[[COND]] then %[[EA]] else %[[EB]] : (!cir.bool, !cir.double, !cir.double) -> !cir.double
  // CIR: %[[IDX0_INS:.+]] = cir.const #cir.int<0> : !u64i
  // CIR: %{{.+}} = cir.vec.insert %[[SEL]], %[[A]][%[[IDX0_INS]] : !u64i] : !cir.vector<2 x !cir.double>

  // LLVM-LABEL: @test_selectsd_128
  // LLVM: %[[E1:.+]] = extractelement <2 x double> %{{.+}}, i64 0
  // LLVM: %[[E2:.+]] = extractelement <2 x double> %{{.+}}, i64 0
  // LLVM: %[[MASK:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // LLVM: %[[BIT:.+]] = extractelement <8 x i1> %[[MASK]], i64 0
  // LLVM: %[[CMP:.+]] = icmp ne i1 %[[BIT]], false
  // LLVM: %[[SEL:.+]] = select i1 %[[CMP]], double %[[E1]], double %[[E2]]
  // LLVM: insertelement <2 x double> %{{.+}}, double %[[SEL]], i64 0

  // OGCG-LABEL: @test_selectsd_128
  // OGCG: %[[E1:.+]] = extractelement <2 x double> %{{.+}}, i64 0
  // OGCG: %[[E2:.+]] = extractelement <2 x double> %{{.+}}, i64 0
  // OGCG: %[[MASK:.+]] = bitcast i8 %{{.+}} to <8 x i1>
  // OGCG: %[[BIT:.+]] = extractelement <8 x i1> %[[MASK]], i64 0
  // OGCG: %{{.+}} = select i1 %[[BIT]], double %[[E1]], double %[[E2]]
  // OGCG: insertelement <2 x double> %{{.+}}, double %{{.+}}, i64 0
  return __builtin_ia32_selectsd_128(k, (__v2df)a, (__v2df)b);
}
