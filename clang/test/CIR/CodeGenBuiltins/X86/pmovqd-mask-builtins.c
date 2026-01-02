// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512bw -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512bw -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512bw -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

#include <immintrin.h>

__m256i test_pmovqd_mask(__m512i a, __m256i b, __mmask8 mask) {
  // CIR-LABEL: test_pmovqd_mask
  // CIR: %[[TRUNC:.*]] = cir.cast integral {{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<8 x !s32i>
  // CIR: %[[MASK_VEC:.*]] = cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary(%[[MASK_VEC]], %[[TRUNC]], {{.*}}) : !cir.vector<8 x !cir.int<s, 1>>, !cir.vector<8 x !s32i>
  // LLVM-LABEL: @test_pmovqd_mask
  // LLVM: %[[B_CAST:.*]] = bitcast <4 x i64> %{{.*}} to <8 x i32>
  // LLVM: %[[TRUNC:.*]] = trunc <8 x i64> %{{.*}} to <8 x i32>
  // LLVM: %[[MASK_VEC:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // LLVM: %[[CMP:.*]] = icmp ne <8 x i1> %[[MASK_VEC]], zeroinitializer
  // LLVM: %[[SEL:.*]] = select <8 x i1> %[[CMP]], <8 x i32> %[[TRUNC]], <8 x i32> %[[B_CAST]]
  // LLVM: %[[RETBC:.*]] = bitcast <8 x i32> %[[SEL]] to <4 x i64>
  // LLVM: store <4 x i64> %[[RETBC]],
  // LLVM: %[[RET:.*]] = load <4 x i64>,
  // LLVM: ret <4 x i64> %[[RET]]
  // OGCG-LABEL: @test_pmovqd_mask
  // OGCG: %[[B_CAST:.*]] = bitcast <4 x i64> %{{.*}} to <8 x i32>
  // OGCG: %[[TRUNC:.*]] = trunc <8 x i64> %{{.*}} to <8 x i32>
  // OGCG: %[[MASK_VEC:.*]] = bitcast i8 %{{.*}} to <8 x i1>
  // OGCG: %[[SEL:.*]] = select <8 x i1> %[[MASK_VEC]], <8 x i32> %[[TRUNC]], <8 x i32> %[[B_CAST]]
  // OGCG: %[[RET:.*]] = bitcast <8 x i32> %[[SEL]] to <4 x i64>
  // OGCG: ret <4 x i64> %[[RET]]
  return __builtin_ia32_pmovqd512_mask(a, b, mask);
}

__m256i test_pmovwb_mask(__m512i a, __m256i b, __mmask32 mask) {
  // CIR-LABEL: test_pmovwb_mask
  // CIR: %[[TRUNC:.*]] = cir.cast integral {{.*}} : !cir.vector<32 x !s16i> -> !cir.vector<32 x !s8i>
  // CIR: %[[MASK_VEC:.*]] = cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.vec.ternary(%[[MASK_VEC]], %[[TRUNC]], {{.*}}) : !cir.vector<32 x !cir.int<s, 1>>, !cir.vector<32 x !s8i>
  // LLVM-LABEL: @test_pmovwb_mask
  // LLVM: %[[A_CAST:.*]] = bitcast <8 x i64> %{{.*}} to <32 x i16>
  // LLVM: %[[B_CAST:.*]] = bitcast <4 x i64> %{{.*}} to <32 x i8>
  // LLVM: %[[TRUNC:.*]] = trunc <32 x i16> %[[A_CAST]] to <32 x i8>
  // LLVM: %[[MASK_VEC:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: %[[CMP:.*]] = icmp ne <32 x i1> %[[MASK_VEC]], zeroinitializer
  // LLVM: %[[SEL:.*]] = select <32 x i1> %[[CMP]], <32 x i8> %[[TRUNC]], <32 x i8> %[[B_CAST]]
  // LLVM: %[[RETBC:.*]] = bitcast <32 x i8> %[[SEL]] to <4 x i64>
  // LLVM: store <4 x i64> %[[RETBC]],
  // LLVM: %[[RET:.*]] = load <4 x i64>,
  // LLVM: ret <4 x i64> %[[RET]]
  // OGCG-LABEL: @test_pmovwb_mask
  // OGCG: %[[A_CAST:.*]] = bitcast <8 x i64> %{{.*}} to <32 x i16>
  // OGCG: %[[B_CAST:.*]] = bitcast <4 x i64> %{{.*}} to <32 x i8>
  // OGCG: %[[TRUNC:.*]] = trunc <32 x i16> %[[A_CAST]] to <32 x i8>
  // OGCG: %[[MASK_VEC:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: %[[SEL:.*]] = select <32 x i1> %[[MASK_VEC]], <32 x i8> %[[TRUNC]], <32 x i8> %[[B_CAST]]
  // OGCG: %[[RET:.*]] = bitcast <32 x i8> %[[SEL]] to <4 x i64>
  // OGCG: ret <4 x i64> %[[RET]]
  return __builtin_ia32_pmovwb512_mask(a, b, mask);
}